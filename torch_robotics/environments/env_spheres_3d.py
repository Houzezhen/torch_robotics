import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiSphereField
from torch_robotics.robots import RobotPointMass, RobotPanda, RobotUr10, RobotPandaCar
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


class EnvSpheres3D(EnvBase):

    def __init__(self, name='EnvDense2D', tensor_args=None, **kwargs):
        spheres = MultiSphereField(torch.tensor([
                    [1, 3.4, 0.2],
                    [1.45, 8, 0.3],
                    [2.0, 6.0, 0.6],
                    [2.55, 0.45, 0.5],
                    [0.65, 0.45, 0.15],
                    [1.75, 0.5, 0.25],
                    [2.3, 5.0, 0.7],
                    [2.5, 1.0, 0.3],
                    [0.1, 2.4, 1.0],
                    [1.1, 3.9, 0.35],
                    [1.6, 1.9, 0.1],
                    [1.3, 5.9, 0.1],

                    ]),
                torch.tensor([
                    0.25,
                    0.15,
                    0.25,
                    0.25,
                    0.25,
                    0.15,
                    0.15,
                    0.15,
                    0.15,
                    0.15,
                    0.2,
                    0.25
                ]),
                tensor_args=tensor_args)

        obj_field = ObjectField([spheres], 'spheres')
        obj_list = [obj_field]

        super().__init__(
            name=name,
            limits=torch.tensor([[0, 0, -0.3], [3.62, 8.2, 1]], **tensor_args),  # environments limits   前俩下标数值为平面限制限制，用于边界碰撞检测
            obj_fixed_list=obj_list,
            tensor_args=tensor_args,
            **kwargs
        )

    def get_gpmp2_params(self, robot=None):
        params = dict(
            opt_iters=300,
            num_samples=64,
            sigma_start=1e-3,
            sigma_gp=1e-1,#1e-1
            sigma_goal_prior=1e-3,
            sigma_coll=5e-2,
            step_size=0.002,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.1,
            sigma_start_sample=1e-3,
            sigma_goal_sample=1e-3,
            solver_params={
                'delta': 0.01,
                'trust_region': True,
                'sparse_computation': False,
                'sparse_computation_block_diag': False,
                'method': 'cholesky',
                # 'method': 'cholesky-sparse',
                #'method': 'inverse',
            },
            stop_criteria=0.1,#0.1
        )
        if isinstance(robot, (RobotPanda,RobotUr10,RobotPandaCar)):
            return params
        else:
            raise NotImplementedError

    def get_rrt_connect_params(self, robot=None):
        params = dict(
            n_iters=10000,
            step_size=torch.pi/30,
            n_radius=torch.pi/4,
            n_pre_samples=50000,

            max_time=180
        )
        if isinstance(robot, (RobotPanda,RobotUr10,RobotPandaCar)):
            return params
        else:
            raise NotImplementedError

if __name__ == '__main__':
    env = EnvSpheres3D(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=DEFAULT_TENSOR_ARGS
    )
    fig, ax = create_fig_and_axes(env.dim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()