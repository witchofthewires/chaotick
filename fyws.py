import numpy as np
from manim import *
import math

class RealtimePlot(Scene):

    def construct(self):
        axes = Axes(x_range=[-10, 10], y_range=[-2, 2])
        t = ValueTracker(0)     
        curve = always_redraw(lambda: axes.plot(
            lambda x: np.sin(x - t.get_value()), 
            x_range=[0, 10], 
            color=PINK
        ))

        self.add(axes, curve)
        self.play(t.animate.set_value(2*math.pi), run_time=3, rate_func=linear)
