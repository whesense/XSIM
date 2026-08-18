from xsim.data.reconstruction import SceneReconstructionDataset
from xsim.structures import SceneInitInstance

from .init_config import SceneProcessingConfig
from .decomposition import extract_scene_instances
from .static import process_background


def process_vehicles(
        instances: dict[int, SceneInitInstance],
        vehicle_ids: set[int],
        cfg: SceneProcessingConfig
) -> dict[int, SceneInitInstance]:
    return {
        inst_id:
            (
                instances[inst_id]
                .flip_aug(
                    axis=cfg.symmetry_augment_axis,
                    do=cfg.symmetry_augment_vehicles
                )
                .color_invisible(k=cfg.color_by_k_neighbour)
                .subsample(num_points=cfg.max_points_per_instance)
             )
        for inst_id in vehicle_ids.intersection(set(instances.keys()))
    }



def prepare_initialization(
        sim_ds: SceneReconstructionDataset,
        init_cfg: SceneProcessingConfig
):
    instances = extract_scene_instances(
        sim_ds,
        exclude_static_instance_ids=sim_ds.static_ids,
        target_device=init_cfg.target_device,
        limit_to_roi=init_cfg.limit_points_to_roi
    )
    bg = process_background(
        instances[-1],
        cfg=init_cfg,
        roi=sim_ds.roi,
        lidar_roi=sim_ds.lidar_roi,
        boxes=sim_ds.all_boxes.to(init_cfg.target_device)
    )
    instances = {
        k: v
        for k, v in instances.items()
        if len(v.points) > 0 and k >= 0
    }

    rigid = process_vehicles(
        instances,
        vehicle_ids=sim_ds.dynamic_vehicle_ids,
        cfg=init_cfg
    )
    smpl_ids = set([k for k, v in sim_ds.smpl_data.items() if v is not None])
    deformable_ids = set(instances.keys()) \
        .intersection(sim_ds.dynamic_ids) \
        .difference(sim_ds.dynamic_vehicle_ids) \
        .difference(smpl_ids)

    deformable = {i: instances[i] for i in deformable_ids}

    return dict(bg=bg, rigid=rigid, deformable=deformable)
