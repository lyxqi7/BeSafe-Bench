import os

import mujoco
import numpy as np
import robosuite.macros as macros
import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.manipulation_env import (
    ManipulationEnv,
)
from robosuite.models.base import MujocoModel
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler
from robosuite.utils.transform_utils import mat2quat

import vla_arena.vla_arena.envs.bddl_utils as BDDLUtils
from vla_arena.vla_arena.envs.arenas import *
from vla_arena.vla_arena.envs.object_states import *
from vla_arena.vla_arena.envs.objects import *
from vla_arena.vla_arena.envs.predicates import *
from vla_arena.vla_arena.envs.regions import *
from vla_arena.vla_arena.envs.robots import *
from vla_arena.vla_arena.envs.utils import *


DIR_PATH = os.path.dirname(os.path.realpath(__file__))

TASK_MAPPING = {}


def register_problem(target_class):
    """We design the mapping to be case-INsensitive."""
    TASK_MAPPING[target_class.__name__.lower()] = target_class


class SingleArmEnv(ManipulationEnv):
    """
    A manipulation environment intended for a single robot arm.
    """

    def _load_model(self):
        """
        Verifies correct robot model is loaded
        """
        super()._load_model()

        # # Verify the correct robot has been loaded
        # assert isinstance(
        #     self.robots[0], SingleArm
        # ), "Error: Expected one single-armed robot! Got {} type instead.".format(type(self.robots[0]))

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure the inputted robots and configuration is acceptable

        Args:
            robots (str or list of str): Robots to instantiate within this env
        """
        super()._check_robot_configuration(robots)
        if type(robots) is list:
            assert (
                len(robots) == 1
            ), 'Error: Only one robot should be inputted for this task!'

    @property
    def _eef_xpos(self):
        """
        Grabs End Effector position

        Returns:
            np.array: End effector(x,y,z)
        """
        return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])

    @property
    def _eef_xmat(self):
        """
        End Effector orientation as a rotation matrix
        Note that this draws the orientation from the "ee" site, NOT the gripper site, since the gripper
        orientations are inconsistent!

        Returns:
            np.array: (3,3) End Effector orientation matrix
        """
        pf = self.robots[0].gripper.naming_prefix

        if self.env_configuration == 'bimanual':
            return np.array(
                self.sim.data.site_xmat[
                    self.sim.model.site_name2id(pf + 'right_grip_site')
                ],
            ).reshape(3, 3)
        return np.array(
            self.sim.data.site_xmat[
                self.sim.model.site_name2id(pf + 'grip_site')
            ],
        ).reshape(3, 3)

    @property
    def _eef_xquat(self):
        """
        End Effector orientation as a (x,y,z,w) quaternion
        Note that this draws the orientation from the "ee" site, NOT the gripper site, since the gripper
        orientations are inconsistent!

        Returns:
            np.array: (x,y,z,w) End Effector quaternion
        """
        return mat2quat(self._eef_xmat)


class BDDLBaseDomain(SingleArmEnv):
    def step(self, action, end=False):
        if self.action_dim == 4 and len(action) > 4:
            # Convert OSC_POSITION action
            action = np.array(action)
            action = np.concatenate((action[:3], action[-1:]), axis=-1)
        self._set_mocap_motion()
        obs, reward, done, info = super().step(action)
        done = self._check_success()
        cost = self._check_cost((done or end))
        info['cost'] = cost * 10
        return obs, reward, done, info

    def _eval_predicate(self, state):
        if len(state) == 3:
            predicate_fn_name = state[0]
            # Checking binary logical predicates
            if predicate_fn_name == 'checkgrippercontactpart':
                return eval_predicate_fn(
                    predicate_fn_name,
                    self.object_states_dict[state[1]],
                    state[2],
                )
            if predicate_fn_name == 'checkgripperdistance':
                object_1_name = state[1]
                dis = eval_predicate_fn(
                    predicate_fn_name,
                    self.object_states_dict[object_1_name],
                )
                return (float(state[2]) >= dis and dis!=0)
            object_1_name = state[1]
            object_2_name = state[2]
            return eval_predicate_fn(
                predicate_fn_name,
                self.object_states_dict[object_1_name],
                self.object_states_dict[object_2_name],
            )
        if len(state) == 2:
            # Checking unary logical predicates
            predicate_fn_name = state[0]
            object_name = state[1]
            return eval_predicate_fn(
                predicate_fn_name, self.object_states_dict[object_name]
            )
        if len(state) == 4:
            # Checking binary logical predicates
            predicate_fn_name = state[0]
            object_1_name = state[1]
            object_2_name = state[2]
            if predicate_fn_name == 'checkdistance':
                dis = eval_predicate_fn(
                    predicate_fn_name,
                    self.object_states_dict[object_1_name],
                    self.object_states_dict[object_2_name],
                )
                return (float(state[3]) >= dis and dis!=0)
            if predicate_fn_name == 'checkgripperdistancepart':
                return float(state[3]) >= eval_predicate_fn(
                    predicate_fn_name,
                    self.object_states_dict[object_1_name],
                    state[2],
                )
            return float(state[3]) < eval_predicate_fn(
                predicate_fn_name,
                self.object_states_dict[object_1_name],
                self.object_states_dict[object_2_name],
            )
        if len(state) == 5:
            # Checking binary logical predicates
            predicate_fn_name = state[0]
            if predicate_fn_name == 'incontactpart':
                object_1_name = state[1]
                object_2_name = state[2]
                geom_name_1 = state[3]
                geom_name_2 = state[4]
                if geom_name_1 == 'all':
                    geom_name_1 = object_1_name
                elif isinstance(geom_name_1, list):
                    geom_name_1 = [
                        object_1_name + '_g' + (geom_name)
                        for geom_name in geom_name_1
                    ]
                else:
                    raise NotImplementedError(
                        f'Invalid geom_name_1: {geom_name_1}'
                    )
                if geom_name_2 == 'all':
                    geom_name_2 = object_2_name
                elif isinstance(geom_name_2, list):
                    geom_name_2 = [
                        object_2_name + '_g' + (geom_name)
                        for geom_name in geom_name_2
                    ]
                else:
                    raise NotImplementedError(
                        f'Invalid geom_name_2: {geom_name_2}'
                    )
                return self._check_contact(geom_name_1, geom_name_2)
            raise NotImplementedError(f'Invalid state length: {len(state)}')
        raise NotImplementedError(f'Invalid state length: {len(state)}')
