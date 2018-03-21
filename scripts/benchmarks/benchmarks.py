from scripts.benchmarks import Cube, Ball, Simplex, BenchmarkModel

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


def generate_model(model_type: BenchmarkModel.BenchmarkModel, i=2, d=2.7, rows=5000):
    model = model_type(i=i, d=d, rows=rows)
    df = model.generate_df()
    model.save(df)


generate_model(model_type=Cube.Cube)
generate_model(model_type=Ball.Ball)
generate_model(model_type=Simplex.Simplex)