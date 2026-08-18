from typing import Optional

import torch
from toast import quat_std, sample_trajectory, use_backend, velocities, LinearMotion, SE3

from xsim.data import SceneReconstructionDataset
from xsim.modeling.scene import Scene
from xsimgs.structures import ROIBox3D, OrientedBoxes

from ..common import SceneNode, OptimizationConfig, init_optimization, LossesType

from xsim.modeling.nodes.gaussian.utils.smpl_node_utils import smpl_boxes_pose_corrected


class TrajectoryNode(SceneNode):
    @staticmethod
    def create(
            sim_ds: SceneReconstructionDataset,
            init: dict,
            optimize: bool = True,
            opt_cfg: Optional[OptimizationConfig] = None,
            losses: LossesType = None
    ) -> "TrajectoryNode":
        subset_mask = torch.isin(
            sim_ds.instances.sensor_link.key(),
            init['train_split'].image_keys.to(sim_ds.device)
        ).squeeze(-1)
        instances: OrientedBoxes = sim_ds.instances.clone()
        assert instances.mask is not None

        inst_q = instances.motion.pose.q.std().data.clone()
        inst_t = instances.motion.pose.t.clone()

        if sim_ds.num_smpl_objects > 0:
            smpl_init = smpl_boxes_pose_corrected(sim_ds)
            smpl_ids = smpl_init['smpl_ids'].to(inst_q.device)
            inst_q[smpl_ids] = smpl_init['smpl_box_q'].to(inst_q.device)
            inst_t[smpl_ids] = smpl_init['smpl_box_t'].to(inst_q.device)

        return TrajectoryNode(
            roi=sim_ds.roi,
            instance_time=instances.motion.pose_time.squeeze(-1),
            instance_q=inst_q,
            instance_t=inst_t,
            instance_mask=instances.mask & subset_mask,
            instance_size=sim_ds.instance_size.clone(),
            optimize=optimize,
            opt_cfg=opt_cfg,
            losses=losses or [],
        )

    def __init__(
            self,
            roi: ROIBox3D,
            instance_time: torch.Tensor,
            instance_q: torch.Tensor,
            instance_t: torch.Tensor,
            instance_mask: torch.Tensor,
            instance_size: torch.Tensor,
            optimize: bool = True,
            max_extrapolation: float = 0.5,
            opt_cfg: Optional[OptimizationConfig] = None,
            losses: LossesType = None
    ):
        super().__init__(roi=roi, losses=losses)
        self.optimize = optimize
        self.max_extrapolation = max_extrapolation
        self.opt_cfg = opt_cfg or OptimizationConfig()

        self.instance_time = torch.nn.Buffer(instance_time)  # [instances, keyframes]
        self.instance_mask = torch.nn.Buffer(instance_mask)  # [instances, keyframes]
        self.instance_q = torch.nn.Buffer(instance_q)  # [instances, keyframes, 4]
        self.instance_t = torch.nn.Buffer(instance_t)  # [instances, keyframes, 3]
        self.instance_size = torch.nn.Buffer(instance_size)  # [instances, 3]
        self.instance_ids = torch.nn.Buffer(
            torch.arange(self.num_objects, device=instance_mask.device, dtype=torch.int32)
        )

        if self.optimize:
            self.instance_q_adj = torch.nn.Parameter(torch.zeros_like(self.instance_q))
            self.instance_t_adj = torch.nn.Parameter(torch.zeros_like(self.instance_t))

    def forward(self, scene: Scene):
        if scene.world_time is not None:
            tmin = tmax = t_middle = torch.tensor(
                scene.world_time, device=self.instance_mask.device)
        else:
            tmin: torch.Tensor = None
            tmax: torch.Tensor = None
            for render in scene.renders:
                tmin_cur, tmax_cur = render.camera.shutter.time_range
                tmin = torch.minimum(tmin_cur, tmin) if tmin is not None else tmin_cur
                tmax = torch.maximum(tmax_cur, tmax) if tmax is not None else tmax_cur
            t_middle = (tmin + tmax) * 0.5

        scene.render_data['t_min'] = tmin
        scene.render_data['t_max'] = tmax
        scene.world_time = t_middle

        query_t = torch.stack([tmin, t_middle, tmax], dim=-1)
        query_t = query_t.view(1, 3).expand(self.num_objects, -1)

        inst_q = self.instance_q
        inst_t = self.instance_t
        if self.optimize:
            inst_q = quat_std(inst_q + self.instance_q_adj)
            inst_t = inst_t + self.instance_t_adj

        (out_q, out_t, out_mask, out_v, out_w, out_indices) = sample_trajectory(
            query_times=query_t,
            seq_times=self.instance_time,
            seq_quats=inst_q,
            seq_t=inst_t,
            seq_mask=self.instance_mask,
            use_only_valid_keyframes=True,
            extrapolate=True,
            extrapolation_window=1,
            mask_mode='and',
            return_velocities=True,
            return_mask=True,
            return_indices=True
        )  # out_q = [objects, 3, 4]

        ind_min = out_indices[:, 0, 0]
        ind_max = out_indices[:, 2, 1]
        ind_min_time = self.instance_time.take_along_dim(
            ind_min.long().view(-1, 1), dim=1).view(-1)
        ind_max_time = self.instance_time.take_along_dim(
            ind_max.long().view(-1, 1), dim=1).view(-1)
        extrapolate_mask = (
                (tmin < ind_min_time - self.max_extrapolation) |
                (ind_max_time + self.max_extrapolation < tmax)
        )

        cur_q, cur_t = out_q[:, 1], out_t[:, 1]
        cur_mask = out_mask[:, 1]

        time_eps = 0.001
        if abs((tmax - tmin).item()) < time_eps:
            # difference between capture start and finish is nearly zero,
            # consider sensors as global shutter, use middle query
            cur_v, cur_w = out_v[:, 1], out_w[:, 1]
        else:
            # use numerical difference to estimate velocities
            cur_v, cur_w = velocities(
                out_q[:, 0], out_t[:, 0],
                out_q[:, 2], out_t[:, 2],
                query_t[:, 0:1], query_t[:, 2:3]
            )

        scene.objects = OrientedBoxes(
            motion=LinearMotion(
                pose=SE3(cur_q, cur_t),
                pose_time=t_middle.view(1, 1).expand(-1, 1).contiguous(),
                linear_velocity=cur_v,
                angular_velocity=cur_w,
            ),
            sizes=self.instance_size,
            object_ids=self.instance_ids,
            mask=cur_mask & ~extrapolate_mask,
        ).contiguous()

        # out_indices: [objects, queries, 2] keyframe indices bracketing each query;
        # these are the keyframes whose pose adjustment received gradient this step.
        return {
            'out_indices': out_indices,
            'mask': scene.objects.mask,
        }

    def remap_instances(self, remaining_instances):
        self.instance_time = self.instance_time[remaining_instances].contiguous()
        self.instance_mask = self.instance_mask[remaining_instances].contiguous()
        self.instance_size = self.instance_size[remaining_instances].contiguous()
        self.instance_q = self.instance_q[remaining_instances].contiguous()
        self.instance_t = self.instance_t[remaining_instances].contiguous()

        if self.optimize:
            self.instance_q_adj = torch.nn.Parameter(
                self.instance_q_adj[remaining_instances].contiguous())
            self.instance_t_adj = torch.nn.Parameter(
                self.instance_t_adj[remaining_instances].contiguous())

    @property
    def num_objects(self):
        return self.instance_mask.shape[0]

    @property
    def num_keyframes(self):
        return self.instance_mask.shape[1]

    def init_optimization(self):
        if not self.optimize:
            return

        self.opt, self.schedulers = init_optimization(
            self.opt_cfg,
            {'pose_quat': self.instance_q_adj, 'pose_t': self.instance_t_adj},
            self.resolve_lr_scale_factor, self.__class__.__name__
        )
