import numpy as np
import jax.numpy as jnp
from jax.numpy.linalg import norm as jnorm

def backtrack(F, tuner, g, x, x_new, fx, fx_new, fx_norm, fx_norm_new, max_iter=5):
    print("backtracking")
    dx = jnp.array(x_new) - jnp.array(x)
    step_len = 0.5
    x_curr = x_new
    fx_curr = fx_new
    fx_norm_curr = fx_norm_new
    fx_norm_old = None
    for i in range(max_iter):
        print(f"fx_norm_curr: {fx_norm_curr}, fx_norm: {fx_norm}")
        if fx_norm_curr < fx_norm:
            return x_curr, fx_curr, fx_norm_curr
        step_len = step_len / 2.0
        x_curr = x + step_len * dx
        x_curr = jnp.array(x_curr)
        x_curr = jnp.where(x_curr < 0, 0.0, x_curr)
        fx_curr = F(np.asarray(x_curr), tuner, g)
        fx_norm_curr = jnorm(fx_curr, ord=2) / len(fx_curr)
        if fx_norm_old is not None and fx_norm_curr > fx_norm_old:
            print("Backtrack failed, likely wrong direction, returning previous value")
            break
        fx_norm_old = fx_norm_curr
    if i == max_iter - 1:
        print("Max backtrack iterations reached, returning previous value")
    return x_curr, fx_curr, fx_norm_curr

def broyden(F, x0, tuner, g, J=None, tol=0.01, max_iter=100):
    if J is None:
        J = jnp.eye(len(x0)) * -1
    J_inv = jnp.linalg.inv(J)
    fx = F(x0, tuner, g)
    x = x0
    old_norm = jnorm(fx, ord=2) / len(fx)
    all_norms = [old_norm]
    for i in range(max_iter):
        x_new = x - (J_inv @ fx)
        x_new = jnp.array(x_new)
        x_new = jnp.where(x_new < 0, 0.0, x_new)
        fx_new = F(np.asarray(x_new), tuner, g)
        norm = jnorm(fx_new, ord=2) / len(fx_new)
        if i > 0 and norm > old_norm:
            x_new, fx_new, norm = backtrack(F, tuner, g, x, x_new, fx, fx_new, old_norm, norm)
        xstep = jnp.array(x_new-x)
        fxstep = jnp.array(fx_new -fx)
        if norm < tol:
            print(all_norms)
            for x in x_new:
                print(x)
            return x_new
        fxstep = fxstep - J @ xstep
        dx2 = jnp.dot(xstep, xstep)
        fxstep = fxstep / dx2
        J_new = J + fxstep * jnp.transpose(xstep)
        J_inv = jnp.linalg.inv(J_new)
        x = x_new
        fx = fx_new
        J = J_new
        all_norms.append(norm)
        old_norm = norm
    print(all_norms)
    raise Exception(f"Did not converge within {max_iter} iterations")