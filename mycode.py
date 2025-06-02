from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

dataset = pd.read_csv('airfoil_results.csv')
# Inputs and outputs
X = dataset[['M', 'P', 'T']]
y = dataset[['CL', 'CD']]  # or use L/D, etc.

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict & verify performance
preds = model.predict(X_test)



class AirfoilOptimization(ElementwiseProblem):
    def __init__(self,model):
        # n_var = 3 input variables (m, p, t)
        # n_obj = 2 objectives (maximize CL, minimize CD)
        # n_constr = 2 constraint (CL >= 0.7) (Cd <= 0.05)
        super().__init__(n_var=3, n_obj=2, n_constr=4, xl=[1, 1, 5], xu=[9, 9, 15])
        self.model = model
    def _evaluate(self, x, out, *args, **kwargs):
        m,p,t = x
        input_df = pd.DataFrame([[m, p, t]], columns=['M', 'P', 'T'])
        cl, cd = self.model.predict(input_df)[0]
        # Objectives: minimize -CL (== maximize CL), and minimize CD
        f1 = -cl
        f2 = cd

        # constraints
        g1 = 1.0 - cl
        g2 = 12.0 - t 
        g3 = 4 - p
        g4 = 3 - m

        out["F"] = [f1, f2]
        out["G"] = [g1, g2,g3,g4]

problem = AirfoilOptimization(model)

algorithm  = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               seed=1,
               verbose=False)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()

# Input variables that gave Pareto front solutions
input_points = res.X

# Print them out
for i, (m, p, t) in enumerate(input_points):
    cl = -res.F[i][0]
    cd = res.F[i][1]
    print(f"Design {i+1}: M={m:.2f}, P={p:.2f}, T={t:.2f} → CL={cl:.3f}, CD={cd:.4f}")
