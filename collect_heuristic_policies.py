import json
import numpy as np

bad_cats = [0, 3, 5, 255]


def min_area(w, h):
    denom = max(1, w * h)
    return 1 / denom


def max_area(w, h):
    return w * h


def weight_detections(
    detections: dict, img_size: tuple[int], size_norm=None
) -> np.ndarray:
    w, h = img_size

    counts = {
        "left_upper_corner": 0,
        "left_lower_corner": 0,
        "right_upper_corner": 0,
        "right_lower_corner": 0,
    }

    for v in detections.values():
        if v["category_id"] not in bad_cats:
            box = v["bbox"]
            x_c = abs(box[0]) + box[2] // 2
            y_c = abs(box[1]) + box[3] // 2

            if size_norm is not None:
                coeff = size_norm(box[2], box[3])
            else:
                coeff = 1

            if x_c < w // 2:
                if y_c < h // 2:
                    counts["left_upper_corner"] += 1 * coeff
                else:
                    counts["left_lower_corner"] += 1 * coeff
            else:
                if y_c < h // 2:
                    counts["right_upper_corner"] += 1 * coeff
                else:
                    counts["right_lower_corner"] += 1 * coeff

    return np.array(
        [
            [counts["left_upper_corner"], counts["right_upper_corner"]],
            [counts["left_lower_corner"], counts["right_lower_corner"]],
        ]
    )


def get_action(wd: np.ndarray, actions) -> str:
    up_sum = sum(wd[0])
    down_sum = sum(wd[1])
    left_sum = sum(wd[:, 0])
    right_sum = sum(wd[:, 1])

    horizontal = (left_sum, right_sum)
    vertical = (down_sum, up_sum)

    turn_right = right_sum > left_sum
    look_up = up_sum > down_sum

    horizontal_sum = horizontal[int(turn_right)]
    vertical_sum = vertical[int(look_up)]

    if horizontal_sum > vertical_sum:
        return actions[0 + int(turn_right)]
    else:
        return actions[2 + int(look_up)]


def get_opposite_action(action: str, actions: list[str]):
    idx = actions.index(action)
    if idx > 3:
        return None

    idx_half = idx // 2
    opposite_idx = 2 * idx_half + int(not idx % 2)
    return actions[opposite_idx]


def h_turn_right_only(
    scene_info: dict,
    root: str,
    actions: list[str],
    img_size: tuple[int],
    size_norm=None,
) -> tuple[str]:
    return [
        "turn_right_waypoint",
        "turn_right_waypoint",
        "turn_right_waypoint",
        "turn_right_waypoint",
    ]


def h_turn_2_to_more_obj_return_2(
    scene_info: dict,
    root: str,
    actions: list[str],
    img_size: tuple[int],
    size_norm=None,
) -> tuple[str]:
    detections = scene_info[root]["detections"]
    wd = weight_detections(detections, img_size, size_norm)

    up_sum = sum(wd[0])
    down_sum = sum(wd[1])
    left_sum = sum(wd[:, 0])
    right_sum = sum(wd[:, 1])

    horizontal = (left_sum, right_sum)
    vertical = (down_sum, up_sum)

    turn_right = right_sum > left_sum
    look_up = up_sum > down_sum

    horizontal_sum = horizontal[int(turn_right)]
    vertical_sum = vertical[int(look_up)]

    if horizontal_sum > vertical_sum:
        first_action = actions[0 + int(turn_right)]
        second_action = actions[2 + int(look_up)]
        third_action = actions[0 + int(not turn_right)]
        fourth_action = actions[2 + int(not look_up)]
    else:
        first_action = actions[2 + int(look_up)]
        second_action = actions[0 + int(turn_right)]
        third_action = actions[2 + int(not look_up)]
        fourth_action = actions[0 + int(not turn_right)]

    return (first_action, second_action, third_action, fourth_action)


def h_turn_1_1_to_more_obj_return_2(
    scene_info: dict,
    root: str,
    actions: list[str],
    img_size: tuple[int],
    size_norm=None,
) -> tuple[str]:
    root_detections = scene_info[root]["detections"]
    rwd = weight_detections(root_detections, img_size, size_norm)

    first_action = get_action(rwd, actions)
    first_detections = scene_info[scene_info[root]["actions"][first_action]][
        "detections"
    ]
    fwd = weight_detections(first_detections, img_size, size_norm)

    second_action = get_action(fwd, actions)

    third_action = get_opposite_action(first_action, actions)
    fourth_action = get_opposite_action(second_action, actions)

    return (first_action, second_action, third_action, fourth_action)


def h_turn_1_1_1_1_to_more_obj(
    scene_info: dict,
    root: str,
    actions: list[str],
    img_size: tuple[int],
    size_norm=None,
) -> tuple[str]:
    detections = scene_info[root]["detections"]

    actions_ = []
    prev_pic = root
    for _ in range(4):
        wd = weight_detections(detections, img_size, size_norm)
        action = get_action(wd, actions)
        actions_.append(action)
        detections = scene_info[scene_info[prev_pic]["actions"][action]]["detections"]

    return tuple(actions_)


POLICIES = {
    "turn_right_only": h_turn_right_only,
    "turn_2_to_more_obj_return_2": h_turn_2_to_more_obj_return_2,
    "turn_1_1_to_more_obj_return_2": h_turn_1_1_to_more_obj_return_2,
    "turn_1_1_1_1_to_more_obj": h_turn_1_1_1_1_to_more_obj,
    "turn_1_tmo_back_turn_1_tmo_return_1": None,
    "turn_2_tmo_back_return_1": None,
    "turn_1_tmo_back_turn_1_tmo_return_1": None,
}


ALLOWED_ACTIONS = [
    "turn_left_waypoint",
    "turn_right_waypoint",
    "look_down_discrete_to_velocity",
    "look_up_discrete_to_velocity",
    "STRAFE_LEFT",
]


def collect_policies(
    val_anno: dict, policies_names: list[str], size_norm=None, img_dir=".", out_dir="."
):
    '''
    val_anno: original loaded val annotations;
    policies_names: functions [from this file] names to call in order to collect actions sequencies;
    size_norm: function name [min_ares/max_area/None] to assign weights to objects;
    img_dir: path to val images (will be used as a key for the annotations);
    out_dir: where to write collected jsons
    '''
    for p_name in policies_names:
        print(f"Collect {p_name}")

        actions = {}
        for item in val_anno["data"]:
            name = f"{img_dir}/{item['scene_name']}/{item['root']}.jpg"
            content = item["state_table"]

            policy = POLICIES[p_name](
                content,
                item["root"],
                ALLOWED_ACTIONS,
                (480, 640),
                size_norm,
            )

            policy = [
                val_anno["metadata"]["actions"].index(action) for action in policy
            ]

            actions[name] = policy

        fname = (
            f"{out_dir}/{p_name}.json"
            if size_norm is None
            else f"{out_dir}/{p_name}_{size_norm.__name__}.json"
        )
        with open(fname, "w") as file:
            json.dump(actions, file)


def main():
    with open(
        "interactron_ade_4_steps/annotations/interactron_v1_val-Copy1.json", "r"
    ) as file:
        val_anno = json.load(file)

    collect_policies(
        val_anno,
        [p for p, f in POLICIES.items() if f is not None],
        None,
        # min_area,
        # max_area,
        img_dir="/datasets/interactron_ade_4_steps/val",
        out_dir="action_policy_study",
    )


if __name__ == "__main__":
    main()
