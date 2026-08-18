# Contains modified code from OmniRe / drivestudio
# (https://github.com/ziyc/drivestudio, MIT, Copyright (c) 2024 Ziyu Chen).
# See THIRD_PARTY_LICENSES.md.

import os
import torch
import torch.nn.functional as F

from chamferdist.chamfer import knn_points
from toast import Quat, SE3

from xsim.data import SceneReconstructionDataset
from xsim.modeling.gaussian.model.init_utils import build_params
from xsim.structures import SceneInitInstance

from .smpl_template import SMPLTemplate
from xsim.modeling.gaussian import GaussianField as GF, InitContext, ParamSpec


def smpl_model_path():
    if 'SMPL_PATH' in os.environ:
        return os.environ['SMPL_PATH']
    return "smpl_models/SMPL_NEUTRAL.pth"


def get_face_normals(v, f):
    fn = torch.cross(
        v[..., f[:, 2], :] - v[..., f[:, 1], :],
        v[..., f[:, 0], :] - v[..., f[:, 1], :],
        dim=-1)
    return F.normalize(fn, dim=-1), fn.norm(dim=-1) * 0.5


def vertex_normals(v, f, fn=None):
    fn = fn if fn is not None else get_face_normals(v, f)[0]

    vn = torch.zeros_like(v)
    vn.index_add_(-2, f[:, 0], fn)
    vn.index_add_(-2, f[:, 1], fn)
    vn.index_add_(-2, f[:, 2], fn)

    return F.normalize(vn, dim=-1, eps=1e-12)


def init_vqs_smpl(template, sz_factor=0.5, device: str = 'cpu'):
    v, f = template.get_init_vf()
    v = v.to(device)
    f = f.to(device)

    fn, f_area = get_face_normals(v, f)
    uz = vertex_normals(v, f, fn=fn)
    rnd_dir = torch.rand_like(uz)
    ux = F.normalize(torch.cross(uz, rnd_dir, dim=-1), dim=-1)
    uy = F.normalize(torch.cross(uz, ux, dim=-1), dim=-1)
    q = Quat.from_matrix(torch.stack([ux, uy, uz], dim=-1)).std().data

    f_area /= 3.0
    vtx_area = torch.zeros_like(v[..., 0])
    vtx_area.scatter_add_(-1, f[:, 0].view(1, -1).expand(len(vtx_area), -1), f_area)
    vtx_area.scatter_add_(-1, f[:, 1].view(1, -1).expand(len(vtx_area), -1), f_area)
    vtx_area.scatter_add_(-1, f[:, 2].view(1, -1).expand(len(vtx_area), -1), f_area)
    radius = (vtx_area / torch.pi).sqrt().clamp(1e-4, 1 - 1e-4)

    sz = (radius * sz_factor).clamp(1e-4, 1 - 1e-4)
    s = torch.stack([radius, radius, sz], dim=-1)

    return v, q, s


def smpl_knn(x, k: int = 3):
    return knn_points(x, x, K=k, return_nn=False)[1]


def smpl_transforms(template: SMPLTemplate, masked_theta, inst_mask, v_full = None):
    W, A = template(
        masked_theta=F.normalize(masked_theta, dim=-1),
        instances_mask=inst_mask,
        xyz_canonical=v_full if template.use_voxel_deformer else None
    )
    T = torch.einsum("bnj, bjrc -> bnrc", W, A)
    return T[..., :3, :3], T[..., :3, 3]


def joint_quat_mtx(
        sim_ds: SceneReconstructionDataset,
):
    max_num_keyframes = sim_ds.instances.shape[1]
    joint_quats = torch.zeros(sim_ds.num_smpl_objects, max_num_keyframes, 24, 4)
    joint_quats[..., 0] = 1

    for i, smpl_id in enumerate(sorted(sim_ds.smpl_object_ids)):
        cur_joints = sim_ds.smpl_data[smpl_id].joint_q
        if joint_quats.device != cur_joints.device:
            joint_quats = joint_quats.to(cur_joints.device)
        joint_quats[i, :len(cur_joints)] = cur_joints

    return joint_quats.to(sim_ds.device)

def smpl_boxes_pose_corrected(
        sim_ds: SceneReconstructionDataset,
):
    device = sim_ds.device
    smpl_ids = sorted(sim_ds.smpl_object_ids)
    betas = torch.stack([sim_ds.smpl_data[i].betas for i in smpl_ids], dim=0)

    template = SMPLTemplate(
        smpl_model_path=smpl_model_path(),
        num_human=betas.shape[0],
        init_beta=betas,
        cano_pose_type="da_pose",
        use_voxel_deformer=False
    ).to(device)
    init_verts = template.get_init_vf()[0]

    mask = sim_ds.instances.mask[smpl_ids]
    max_num_keyframes = sim_ds.instances.shape[1]
    joint_quats = joint_quat_mtx(sim_ds)

    inst_t_corrections = torch.zeros(
        template.num_human, max_num_keyframes, 3, device=device)

    print('join_quats:', joint_quats.device, 'cur_mask:', mask.device)

    for k in range(max_num_keyframes):
        cur_mask = mask[:, k]
        if cur_mask.sum() == 0:
            continue


        cur_poses = joint_quats[cur_mask, k]
        R, t = smpl_transforms(template, cur_poses, cur_mask)
        deformed_means = torch.einsum("bnij,bnj->bni", R, init_verts[cur_mask]) + t
        bbox_center = (deformed_means.amin(dim=1) + deformed_means.amax(dim=1)) * 0.5
        inst_t_corrections[cur_mask, k] = -bbox_center

    inst_t_corrections = inst_t_corrections.to(sim_ds.instances.device)
    smpl_box_q = joint_quats[:, :, 0]
    smpl_box_t = sim_ds.instances.pose.t[smpl_ids] + inst_t_corrections

    return dict(
        smpl_box_q=smpl_box_q,
        smpl_box_t=smpl_box_t,
        smpl_ids=torch.tensor(smpl_ids, device=device),
        joint_quats=joint_quats
    )


def make_smpl_template(betas, use_voxel_deformer, device):
    template = SMPLTemplate(
        smpl_model_path=smpl_model_path(),
        num_human=betas.shape[0],
        init_beta=betas,
        cano_pose_type="da_pose",
        use_voxel_deformer=use_voxel_deformer
    ).to(device)
    if use_voxel_deformer:
        template.voxel_deformer.enable_voxel_correction()

    return template


def init_smpl_node(
        sim_ds: SceneReconstructionDataset,
        model_type: type,  # gaussian model type returned by compose
        init_configs: list = None,
        use_voxel_deformer: bool = True,
        device: str = 'cpu',
):
    smpl_ids = sorted(sim_ds.smpl_object_ids)
    betas = torch.stack([sim_ds.smpl_data[i].betas for i in smpl_ids], dim=0)
    joint_quats = joint_quat_mtx(sim_ds)

    template = make_smpl_template(betas, use_voxel_deformer, device)
    instance_ids = torch.as_tensor(smpl_ids, device=device, dtype=torch.int32)
    instance_ids = instance_ids.repeat_interleave(template.num_vertices).view(-1, 1)

    x, q, s = init_vqs_smpl(template, device=device)

    specs: dict[str, ParamSpec] = model_type._param_specs
    init_ctx = InitContext(
        instances={
            smpl_id: SceneInitInstance(points=x[i])
            for i, smpl_id in enumerate(smpl_ids)
        },
        device=device,
        built_params={
            GF.positions: specs[GF.positions].inv_act(x.reshape(-1, 3)).contiguous(),
            GF.rotation: specs[GF.rotation].inv_act(q.reshape(-1, 4)).contiguous(),
            GF.scale: specs[GF.scale].inv_act(s.reshape(-1, 3)).contiguous(),
            "instance_ids": instance_ids.contiguous(),
        },
    )
    model_params = dict(params=build_params(model_type, init_ctx, configs=init_configs))

    return template, model_params, joint_quats
