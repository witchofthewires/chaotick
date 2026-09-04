import numpy as np
from manim import *

class RealtimePlot(Scene):

    def construct(self):
        axes = Axes(x_range=[0, 10], y_range=[-2, 2])
        t = ValueTracker(0)     
        curve = always_redraw(lambda: axes.plot(
            lambda x: np.sin(x - t.get_value()), 
            x_range=[0, 10], 
            color=YELLOW
        ))

        self.add(axes, curve)
        self.play(t.animate.set_value(10), run_time=5, rate_func=linear)
