import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16, 'axes.labelweight': 'bold'})




def plot_gradient_m(x, y, m, slopes, mse, grad_func):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=slopes,
            y=mse,
            line_color="#1ac584",
            line=dict(width=3),
            mode="lines",
            name="MSE",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=slopes,
            y=mean_squared_error(y, m * x) + grad_func(x, y, m) * (slopes - m),
            line_color="red",
            mode="lines",
            line=dict(width=2),
            name="gradient",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[m],
            y=[mean_squared_error(y, m * x)],
            line_color="red",
            marker=dict(size=14, line=dict(width=1, color="DarkSlateGrey")),
            mode="markers",
            name=f"slope {m}",
        )
    )
    fig.update_layout(
        width=520,
        height=450,
        xaxis_title="slope (w)",
        yaxis_title="MSE",
        title=f"slope {m:.1f}, gradient {grad_func(x, y, m):.1f}",
        title_x=0.46,
        title_y=0.93,
        margin=dict(t=60),
    )
    fig.update_xaxes(range=[0.4, 1.6], tick0=0.4, dtick=0.2)
    fig.update_yaxes(range=[0, 2500])
    return fig


def plot_grid_search(
    x,
    y,
    slopes,
    loss_function,
    title="Mean Squared Error",
    y_range=[0, 2500],
    y_title="MSE",
):
    mse = []
    df = pd.DataFrame()
    for m in slopes:
        df[f"{m:.2f}"] = m * x  # store predictions for plotting later
        mse.append(loss_function(y, m * x))  # calc MSE
    mse = pd.DataFrame({"slope": slopes, "squared_error": mse})
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Data & Fitted Line", title)
    )
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers", marker=dict(size=10), name="Data"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df.iloc[:, 0],
            line_color="red",
            mode="lines",
            line=dict(width=3),
            name="Fitted line",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=mse["slope"],
            y=mse["squared_error"],
            mode="markers",
            marker=dict(size=7),
            name="MSE",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=mse.iloc[[0]]["slope"],
            y=mse.iloc[[0]]["squared_error"],
            line_color="red",
            mode="markers",
            marker=dict(size=14, line=dict(width=1, color="DarkSlateGrey")),
            name="MSE for line",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(width=900, height=475)
    fig.update_xaxes(
        range=[10, 130],
        tick0=10,
        dtick=20,
        row=1,
        col=1,
        title="defense",
        title_standoff=0,
    )
    fig.update_xaxes(
        range=[0.3, 1.6],
        tick0=0.3,
        dtick=0.2,
        row=1,
        col=2,
        title="slope",
        title_standoff=0,
    )
    fig.update_yaxes(
        range=[10, 130],
        tick0=10,
        dtick=20,
        row=1,
        col=1,
        title="attack",
        title_standoff=0,
    )
    fig.update_yaxes(
        range=y_range, row=1, col=2, title=y_title, title_standoff=0
    )
    frames = [
        dict(
            name=f"{slope:.2f}",
            data=[
                go.Scatter(x=x, y=y),
                go.Scatter(x=x, y=df[f"{slope:.2f}"]),
                go.Scatter(x=mse["slope"], y=mse["squared_error"]),
                go.Scatter(
                    x=mse.iloc[[n]]["slope"], y=mse.iloc[[n]]["squared_error"]
                ),
            ],
            traces=[0, 1, 2, 3],
        )
        for n, slope in enumerate(slopes)
    ]

    sliders = [
        {
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "slope: ",
                "visible": True,
            },
            "pad": {"b": 10, "t": 30},
            "steps": [
                {
                    "args": [
                        [f"{slope:.2f}"],
                        {
                            "frame": {
                                "duration": 0,
                                "easing": "linear",
                                "redraw": False,
                            },
                            "transition": {"duration": 0, "easing": "linear"},
                        },
                    ],
                    "label": f"{slope:.2f}",
                    "method": "animate",
                }
                for slope in slopes
            ],
        }
    ]
    fig.update(frames=frames), fig.update_layout(sliders=sliders)
    return fig


def plot_grid_search_2d(x, y, slopes, intercepts):
    mse = np.zeros((len(slopes), len(intercepts)))
    for i, slope in enumerate(slopes):
        for j, intercept in enumerate(intercepts):
            mse[i, j] = mean_squared_error(y, x * slope + intercept)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Surface Plot", "Contour Plot"),
        specs=[[{"type": "surface"}, {"type": "contour"}]],
    )
    fig.add_trace(
        go.Surface(
            z=mse, x=intercepts, y=slopes, name="", colorscale="viridis"
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Contour(
            z=mse,
            x=intercepts,
            y=slopes,
            name="",
            showscale=False,
            colorscale="viridis",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        scene=dict(
            zaxis=dict(title="MSE"),
            yaxis=dict(title="slope (w<sub>1</sub>)"),
            xaxis=dict(title="intercept (w<sub>0</sub>)"),
        ),
        scene_camera=dict(eye=dict(x=2, y=1.1, z=1.2)),
        margin=dict(l=0, r=0, b=60, t=90),
    )
    fig.update_xaxes(
        title="intercept (w<sub>0</sub>)",
        range=[intercepts.max(), intercepts.min()],
        tick0=intercepts.max(),
        row=1,
        col=2,
        title_standoff=0,
    )
    fig.update_yaxes(
        title="slope (w<sub>1</sub>)",
        range=[slopes.min(), slopes.max()],
        tick0=slopes.min(),
        row=1,
        col=2,
        title_standoff=0,
    )
    fig.update_layout(width=900, height=475, margin=dict(t=60))
    return fig


def plot_gradient_descent(x, y, w, alpha, tolerance=2e-4, max_iterations=5000):
    if x.ndim == 1:
        x = np.array(x).reshape(-1, 1)
    slopes, losses = gradient_descent(
        x, y, [w], alpha, tolerance, max_iterations, history=True
    )
    slopes = [_[0] for _ in slopes]
    x = x.flatten()
    mse = []
    df = pd.DataFrame()
    for w in slopes:
        df[f"{w:.2f}"] = w * x  # store predictions for plotting later
    slope_range = np.arange(0.4, 1.65, 0.05)
    for w in slope_range:
        mse.append(mean_squared_error(y, w * x))  # calc MSE
    mse = pd.DataFrame({"slope": slope_range, "squared_error": mse})

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Data & Fitted Line", "Mean Squared Error"),
    )
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers", marker=dict(size=10), name="Data"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df.iloc[:, 0],
            line_color="red",
            mode="lines",
            line=dict(width=3),
            name="Fitted line",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=mse["slope"],
            y=mse["squared_error"],
            line_color="#1ac584",
            line=dict(width=3),
            mode="lines",
            name="MSE",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=np.array(slopes[:1]),
            y=np.array(losses[:1]),
            line_color="salmon",
            line=dict(width=4),
            marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey")),
            mode="markers+lines",
            name="Slope history",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=np.array(slopes[0]),
            y=np.array(losses[0]),
            line_color="red",
            mode="markers",
            marker=dict(size=18, line=dict(width=1, color="DarkSlateGrey")),
            name="MSE for line",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[30.3],
            y=[120],
            mode="text",
            text=f"<b>Slope {slopes[0]:.2f}<b>",
            textfont=dict(size=16, color="red"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.update_layout(width=900, height=475, margin=dict(t=60))
    fig.update_xaxes(
        range=[10, 130], tick0=10, dtick=20, title="defense", title_standoff=0, row=1, col=1
    ), fig.update_xaxes(range=[0.4, 1.6], tick0=0.4, dtick=0.2, title="slope (w)", title_standoff=0, row=1, col=2)
    fig.update_yaxes(
        range=[10, 130], tick0=10, dtick=20, title="attack", title_standoff=0, row=1, col=1
    ), fig.update_yaxes(range=[0, 2500], title="MSE", title_standoff=0, row=1, col=2)

    frames = [
        dict(
            name=n,
            data=[
                go.Scatter(x=x, y=y),
                go.Scatter(x=x, y=df[f"{slope:.2f}"]),
                go.Scatter(x=mse["slope"], y=mse["squared_error"]),
                go.Scatter(
                    x=np.array(slopes[: n + 1]),
                    y=np.array(losses[: n + 1]),
                    mode="markers" if n == 0 else "markers+lines",
                ),
                go.Scatter(x=np.array(slopes[n]), y=np.array(losses[n])),
                go.Scatter(text=f"<b>Slope {slope:.2f}<b>"),
            ],
            traces=[0, 1, 2, 3, 4, 5],
        )
        for n, slope in enumerate(slopes)
    ]

    sliders = [
        {
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Iteration: ",
                "visible": True,
            },
            "pad": {"b": 10, "t": 30},
            "steps": [
                {
                    "args": [
                        [n],
                        {
                            "frame": {
                                "duration": 0,
                                "easing": "linear",
                                "redraw": False,
                            },
                            "transition": {"duration": 0, "easing": "linear"},
                        },
                    ],
                    "label": n,
                    "method": "animate",
                }
                for n in range(len(slopes))
            ],
        }
    ]
    fig.update(frames=frames), fig.update_layout(sliders=sliders)
    return fig


def plot_gradient_descent_2d(
    x,
    y,
    w,
    alpha,
    m_range,
    b_range,
    tolerance=2e-5,
    max_iterations=5000,
    step_size=1,
    markers=False,
    stochastic=False,
    batch_size=None,
    seed=None,
):
    if x.ndim == 1:
        x = np.array(x).reshape(-1, 1)
    if stochastic:
        if batch_size is None:
            weights, losses = stochastic_gradient_descent(
                np.hstack((np.ones((len(x), 1)), x)),
                y,
                w,
                alpha,
                tolerance,
                max_iterations,
                history=True,
                seed=seed,
            )
            title = "Stochastic Gradient Descent"
        else:
            weights, losses = minibatch_gradient_descent(
                np.hstack((np.ones((len(x), 1)), x)),
                y,
                w,
                alpha,
                batch_size,
                tolerance,
                max_iterations,
                history=True,
                seed=seed,
            )
            title = "Minibatch Gradient Descent"
    else:
        weights, losses = gradient_descent(
            np.hstack((np.ones((len(x), 1)), x)),
            y,
            w,
            alpha,
            tolerance,
            max_iterations,
            history=True,
        )
        title = "Gradient Descent"
    weights = np.array(weights)
    intercepts, slopes = weights[:, 0], weights[:, 1]
    mse = np.zeros((len(m_range), len(b_range)))
    for i, slope in enumerate(m_range):
        for j, intercept in enumerate(b_range):
            mse[i, j] = mean_squared_error(y, x * slope + intercept)

    fig = make_subplots(
        rows=1,
        subplot_titles=[title],  # . Iterations = {len(intercepts) - 1}."],
    )
    fig.add_trace(
        go.Contour(z=mse, x=b_range, y=m_range, name="", colorscale="viridis")
    )
    mode = "markers+lines" if markers else "lines"
    fig.add_trace(
        go.Scatter(
            x=intercepts[::step_size],
            y=slopes[::step_size],
            mode=mode,
            line=dict(width=2.5),
            line_color="coral",
            marker=dict(
                opacity=1,
                size=np.linspace(19, 1, len(intercepts[::step_size])),
                line=dict(width=2, color="DarkSlateGrey"),
            ),
            name="Descent Path",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[intercepts[0]],
            y=[slopes[0]],
            mode="markers",
            marker=dict(size=20, line=dict(width=2, color="DarkSlateGrey")),
            marker_color="orangered",
            name="Start",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[intercepts[-1]],
            y=[slopes[-1]],
            mode="markers",
            marker=dict(size=20, line=dict(width=2, color="DarkSlateGrey")),
            marker_color="yellowgreen",
            name="End",
        )
    )
    fig.update_layout(
        width=700,
        height=600,
        margin=dict(t=60),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    fig.update_xaxes(
        title="intercept (w<sub>0</sub>)",
        range=[b_range.min(), b_range.max()],
        tick0=b_range.min(),
        row=1,
        col=1,
        title_standoff=0,
    )
    fig.update_yaxes(
        title="slope (w<sub>1</sub>)",
        range=[m_range.min(), m_range.max()],
        tick0=m_range.min(),
        row=1,
        col=1,
        title_standoff=0,
    )
    return fig


def plot_logistic(
    x,
    y,
    y_hat=None,
    threshold=None,
    x_range=[-3, 3],
    y_range=[-0.25, 1.25],
    dx=1,
    dy=0.25,
):
    fig = go.Figure()
    fig.update_xaxes(range=x_range, tick0=x_range[0], dtick=dx)
    fig.update_yaxes(range=y_range, tick0=y_range[0], dtick=dy)
    if threshold is not None:
        threshold_ind = (np.abs(y_hat - threshold)).argmin()
        fig.add_trace(
            go.Scatter(
                x=[x_range[0], x_range[0], x[threshold_ind], x[threshold_ind]],
                y=[y_range[0], y_range[1], y_range[1], y_range[0]],
                mode="lines",
                fill="toself",
                fillcolor="limegreen",
                opacity=0.2,
                line=dict(width=0),
                name="0 prediction",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[x[threshold_ind], x[threshold_ind], x_range[1], x_range[1]],
                y=[y_range[0], y_range[1], y_range[1], y_range[0]],
                mode="lines",
                fill="toself",
                fillcolor="lightsalmon",
                opacity=0.3,
                line=dict(width=0),
                name="1 prediction",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=10,
                color="#636EFA",
                line=dict(width=1, color="DarkSlateGrey"),
            ),
            name="data",
        )
    )
    if y_hat is not None:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_hat,
                line_color="red",
                mode="lines",
                line=dict(width=3),
                name="Fitted line",
            )
        )
        width = 650
        title_x = 0.46
    else:
        width = 600
        title_x = 0.5
    if threshold is not None:
        fig.add_trace(
            go.Scatter(
                x=[x[threshold_ind]],
                y=[threshold],
                mode="markers",
                marker=dict(
                    size=18,
                    color="gold",
                    line=dict(width=1, color="DarkSlateGrey"),
                ),
                name="Threshold",
            )
        )
    fig.update_layout(
        width=width,
        height=500,
        title="Pokemon stats",
        title_x=title_x,
        title_y=0.93,
        xaxis_title="defense",
        yaxis_title="legendary",
        margin=dict(t=60),
    )
    return fig




def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gradient(x, y, w):
    """MSE gradient."""
    y_hat = x @ w
    error = y - y_hat
    gradient = -(1.0 / len(x)) * 2 * x.T @ error
    mse = (error ** 2).mean()
    return gradient, mse


def gradient_descent(
    x,
    y,
    w,
    alpha,
    tolerance: float = 2e-5,
    max_iterations: int = 1000,
    verbose: bool = False,
    print_progress: int = 10,
    history: bool = False,
):
    """MSE gradient descent."""
    iterations = 1
    if verbose:
        print(f"Iteration 0.", "Weights:", [f"{_:.2f}" for _ in w])
    if history:
        ws = []
        mses = []
    while True:
        g, mse = gradient(x, y, w)
        if history:
            ws.append(list(w))
            mses.append(mse)
        w_new = w - alpha * g
        if sum(abs(w_new - w)) < tolerance:
            if verbose:
                print(f"Converged after {iterations} iterations!")
                print("Final weights:", [f"{_:.2f}" for _ in w_new])
            break
        if iterations % print_progress == 0:
            if verbose:
                print(
                    f"Iteration {iterations}.",
                    "Weights:",
                    [f"{_:.2f}" for _ in w_new],
                )
        iterations += 1
        if iterations > max_iterations:
            if verbose:
                print(f"Reached max iterations ({max_iterations})!")
                print("Final weights:", [f"{_:.2f}" for _ in w_new])
            break
        w = w_new
    if history:
        w = w_new
        _, mse = gradient(x, y, w)
        ws.append(list(w))
        mses.append(mse)
        return ws, mses


def stochastic_gradient_descent(
    x,
    y,
    w,
    alpha,
    tolerance: float = 2e-5,
    max_iterations: int = 1000,
    verbose: bool = False,
    print_progress: int = 10,
    history: bool = False,
    seed=None,
):
    """MSE stochastic gradient descent."""
    if seed is not None:
        np.random.seed(seed)
    iterations = 1
    if verbose:
        print(f"Iteration 0.", "Weights:", [f"{_:.2f}" for _ in w])
    if history:
        ws = []
        mses = []
    while True:
        i = np.random.randint(len(x))
        g, mse = gradient(x[i, None], y[i, None], w)
        if history:
            ws.append(list(w))
            mses.append(mse)
        w_new = w - alpha * g
        if sum(abs(w_new - w)) < tolerance:
            if verbose:
                print(f"Converged after {iterations} iterations!")
                print("Final weights:", [f"{_:.2f}" for _ in w_new])
            break
        if iterations % print_progress == 0:
            if verbose:
                print(
                    f"Iteration {iterations}.",
                    "Weights:",
                    [f"{_:.2f}" for _ in w_new],
                )
        iterations += 1
        if iterations > max_iterations:
            if verbose:
                print(f"Reached max iterations ({max_iterations})!")
                print("Final weights:", [f"{_:.2f}" for _ in w_new])
            break
        w = w_new
    if history:
        w = w_new
        _, mse = gradient(x, y, w)
        ws.append(list(w))
        mses.append(mse)
        return ws, mses