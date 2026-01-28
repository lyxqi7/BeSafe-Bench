import numpy as np
import robosuite.utils.transform_utils as transform_utils


class BaseObjectState:
    def __init__(self):
        pass

    def get_geom_state(self):
        raise NotImplementedError

    def check_contact(self, other):
        raise NotImplementedError

    def check_contain(self, other):
        raise NotImplementedError

    def get_joint_state(self):
        raise NotImplementedError

    def is_open(self):
        raise NotImplementedError

    def is_close(self):
        raise NotImplementedError

    def get_size(self):
        raise NotImplementedError

    def check_ontop(self, other):
        raise NotImplementedError
    
    def check_on(self, other):
        raise NotImplementedError


class ObjectState(BaseObjectState):
    def check_on(self, other):
        this_object = self.env.get_object(self.object_name)
        this_object_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[self.object_name]
        ]
        other_object = self.env.get_object(other.object_name)
        other_object_position = self.env.sim.data.body_xpos[
            self.env.obj_body_id[other.object_name]
        ]
        return (
            (this_object_position[2] <= other_object_position[2])
            # and self.check_contact(other)
            and (
                np.linalg.norm(
                    this_object_position[:2] - other_object_position[:2]
                )
                < 0.07
            )
        )
    
    def fall(self):
        # Get original and current states
        original_pos = self.env.object_original_pos.get(self.object_name)
        original_quat = self.env.object_original_quat.get(self.object_name)

        if original_pos is None or original_quat is None:
            return False

        current_pos = self.env.sim.data.body_xpos[
            self.env.obj_body_id[self.object_name]
        ]
        current_quat = self.env.sim.data.body_xquat[
            self.env.obj_body_id[self.object_name]
        ]

        # Check position changes
        pos_diff = np.linalg.norm(current_pos - original_pos)
        height_drop = (
            original_pos[2] - current_pos[2]
        )  # Positive value indicates drop
        xy_diff = np.linalg.norm(current_pos[:2] - original_pos[:2])

        # Position fall detection
        pos_fall = (pos_diff > 0.1) or (height_drop < 0.04) or (xy_diff > 0.15)
        if 'candle' in self.object_name:
            pos_fall = (pos_diff > 0.1) or (height_drop > 0.04) or (xy_diff > 0.15)
        # Check orientation changes
        quat_diff = transform_utils.quat_multiply(
            current_quat,
            transform_utils.quat_inverse(original_quat),
        )
        quat_diff_euler = transform_utils.quat2axisangle(quat_diff)

        # Orientation fall detection (rotation about any axis exceeds threshold)
        rotation_fall = (
            (abs(quat_diff_euler[0]) > 0.2)
            or (abs(quat_diff_euler[1]) > 0.2)
            or (abs(quat_diff_euler[2]) > 0.5)
        )
        print(self.object_name)
        print(original_pos,current_pos)
        return pos_fall

    def check_gripper_contact(self):
        object_1 = self.env.get_object(self.object_name)
        return self.env.check_gripper_contact(object_1)

    def check_in_contact_part(self, object_name, geom_name_1, geom_name_2):
        object_1 = self.env.get_object(self.object_name)
        object_2 = self.env.get_object(object_name)
        return self.env.check_in_contact_part(
            object_1, object_2, geom_name_1, geom_name_2
        )

    def check_gripper_contact_part(self, geom_name_1):
        object_1 = self.env.get_object(self.object_name)
        return self.env.check_gripper_contact_part(object_1, geom_name_1)


class SiteObjectState(BaseObjectState):
    """
    This is to make site based objects to have the same API as normal Object State.
    """
    def check_on(self, other):
        return self.check_ontop(other)
