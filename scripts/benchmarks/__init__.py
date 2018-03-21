from scipy.optimize import linprog

# c = [-1, 4]
# A = [[-3, 1], [1, 2]]
# b = [6, 4]
# x0_bounds = (None, None)
# x1_bounds = (-3, None)
#
# res = linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds, x1_bounds), options={"disp": True})
# print(res)

# from scripts.benchmarks import Cube, Ball, Simplex, BenchmarkModel
import matplotlib.pyplot as plt
from scripts.drawer import draw

# cube = Cube.Cube(i=2, d=2.7)
# cube_df = cube.generate_df()
# cube.save(cube_df)
#
# ball = Ball.Ball(i=2, d=2.7)
# ball_df = ball.generate_df()
# ball.save(ball_df)
#
# simplex = Simplex.Simplex(i=2, d=2.7)
# simplex_df = simplex.generate_df()
# simplex.save(simplex_df)


# def generate_model(model_type: BenchmarkModel.BenchmarkModel, i=2, d=2.7):
#     model = model_type(i=i, d=d)
#     df = model.generate_df()
#     model.save(df)
#
#
# generate_model(model_type=Cube.Cube)