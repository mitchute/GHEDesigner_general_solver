import numpy as np
from OpenGL.GL import *
from OpenGL_2D_class_GLFW import gl2D, gl2DCircle, gl2DText,gl2DArrow,gl2DArc
from reading_inputs import load_data


def drawnetwork(system):
    pipes = system.pipes
    nodes = system.nodes
    # Drawing lines/pipes
    glColor3f(0, 0, 1)
    glLineWidth(3)
    glBegin(GL_LINES)  # begin drawing connected lines
    for pipe in pipes:
        glVertex2f(pipe.input.x, pipe.input.y)
        glVertex2f(pipe.output.x, pipe.output.y)
    glEnd()

    # Putting nodes
    glColor3f(1, 0, 0)
    glLineWidth(3)
    radius = 0.5
    for node in nodes:
        gl2DCircle(node.x, node.y, radius, fill=True)

    # Putting arrows
    glColor3f(0, 0, 1)
    glLineWidth(3)
    gl2DArrow(10, 70, size=1, angleDeg=90, widthDeg=30, toCenter=False, fill=True)
    gl2DArrow(30, 70, size=1, angleDeg=0, widthDeg=30, toCenter=False, fill=True)
    gl2DArrow(30, 0, size=1, angleDeg=270, widthDeg=30, toCenter=False, fill=True)
    gl2DArrow(10, 0, size=1, angleDeg=180, widthDeg=30, toCenter=False, fill=True)

    # Putting text
    glColor3f(1,1,1)
    glLineWidth(3)
    gl2DText("3GHE-6HP SYSTEM", 15, -5)


def main():
    # Draw the house, set the window width and height
    gl2d = gl2D(None, drawnetwork, width=2000, height=1500)
    gl2d.setViewSize(-10, 50, -10, 80, False)
    gl2d.glWait()  # wait for the user to close the window

    print("Finished drawing 1")


main()
