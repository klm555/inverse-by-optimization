#%%
import numpy as np
import matplotlib.pyplot as plt

# Some variables
side_length = 40.
num_points = 40

x_cen, y_cen, a, b, gamma = 25., 25., 10., 6., np.pi/3

# Define the boundary points of the void
def ellipse(x_cen, y_cen, a, b, gamma, rotated_ang):
    x = a * np.cos(rotated_ang) * np.cos(gamma) - b * np.sin(rotated_ang) * np.sin(gamma) + x_cen
    y = a * np.cos(rotated_ang) * np.sin(gamma) + b * np.sin(rotated_ang) * np.cos(gamma) + y_cen
    return np.array([x, y]).T

# Define the domain points
# Find the vertices of the domain
vertices = np.array([[0, 0],
                        [side_length, 0],
                        [side_length, side_length],
                        [0, side_length]])
vertices_x, vertices_y = vertices[:, 0], vertices[:, 1]
# Find the intersection points of ellipse perimeter & the line between the center of ellipse and four vertices
ellipse_lhs = (((vertices_x - x_cen) * np.cos(gamma) + (vertices_y - y_cen) * np.sin(gamma))**2 / a**2) +\
    (((vertices_x - x_cen) * np.sin(gamma) - (vertices_y - y_cen) * np.cos(gamma))**2 / b**2)
t = 1 - 1 / np.sqrt(ellipse_lhs)
x_intersect = (1 - t) * vertices_x + t * x_cen
y_intersect = (1 - t) * vertices_y + t * y_cen
intersections = np.stack([x_intersect, y_intersect], axis=1)
# Untranslate & unrotate
x_intersect_untrans = x_intersect - x_cen
y_intersect_untrans = y_intersect - y_cen
x_intersect_unrotat = x_intersect_untrans * np.cos(gamma) + y_intersect_untrans * np.sin(gamma)
y_intersect_unrotat = -x_intersect_untrans * np.sin(gamma) + y_intersect_untrans * np.cos(gamma)
# Find the angles of intersection points
angles_intersect = np.arctan2(y_intersect_unrotat / b, x_intersect_unrotat / a)
angles_intersect = np.sort(angles_intersect)
# Interpolate between angles
angles_interpolate1 = np.linspace(angles_intersect[0], angles_intersect[1], num_points + 2)[1:-1]
angles_interpolate2 = np.linspace(angles_intersect[1], angles_intersect[2], num_points + 2)[1:-1]
angles_interpolate3 = np.linspace(angles_intersect[2], angles_intersect[3], num_points + 2)[1:-1]
angles_interpolate4 = np.linspace(angles_intersect[3], angles_intersect[0] + 2 * np.pi, num_points + 2)[1:-1]
angles_total = np.concatenate((angles_interpolate1, angles_interpolate2, angles_interpolate3, angles_interpolate4))
# Points on the perimeter of ellipse
domain_points_perimeter = ellipse(x_cen, y_cen, a, b, gamma, angles_total)
# Points on the sides of ellipse
points_interpolate = np.linspace(0, side_length, num_points + 2)[1:-1]
domain_points_bot = np.column_stack((points_interpolate, np.zeros_like(points_interpolate)))
domain_points_right = np.column_stack((side_length * np.ones_like(points_interpolate), points_interpolate))
domain_points_top = np.column_stack((points_interpolate[::-1], side_length * np.ones_like(points_interpolate)))
domain_points_left = np.column_stack((np.zeros_like(points_interpolate), points_interpolate[::-1]))
domain_points_sides = np.concatenate((domain_points_right, domain_points_top, domain_points_left, domain_points_bot), axis=0)

domain_points_x = np.linspace(domain_points_perimeter[:, 0], domain_points_sides[:, 0], num_points + 2)[1:-1]
domain_points_y = np.linspace(domain_points_perimeter[:, 1], domain_points_sides[:, 1], num_points + 2)[1:-1]
domain_points = np.column_stack((domain_points_x.ravel(), domain_points_y.ravel()))

plt.figure()
plt.scatter(domain_points[:, 0], domain_points[:, 1], color='blue', label='Domain Points', s=0.5)
box_x = [0, side_length, side_length, 0, 0]
box_y = [0, 0, side_length, side_length, 0]
plt.plot(box_x, box_y, color='black', label='Domain Boundary')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
# Some variables
side_length = 40.
num_points = 40

# Define the domain & external boundary points
lb = np.array([0., 0.]) # lower bound
ub = np.array([side_length, side_length]) # upper bound
boundary_samples = lhs(2, 1000)
boundary_points = lb + (ub - lb) * boundary_samples

x_cen, y_cen, a, b, gamma = 25., 30., 5., 2., np.pi/3

# Define the boundary points of the void
def ellipse(x_cen, y_cen, a, b, gamma, rotated_ang):
    x = a * np.cos(rotated_ang) * np.cos(gamma) - b * np.sin(rotated_ang) * np.sin(gamma) + x_cen
    y = a * np.cos(rotated_ang) * np.sin(gamma) + b * np.sin(rotated_ang) * np.cos(gamma) + y_cen
    return np.column_stack((x, y))
random_angle1 = 2 * np.pi * boundary_samples[:, 0]
random_angle2 = 2 * np.pi * boundary_samples[:, 1]
void_boundary_points = ellipse(x_cen, y_cen, a, b, gamma, random_angle1)



plt.figure()
plt.scatter(void_boundary_points[:, 0], void_boundary_points[:, 1], color='blue', label='Void Boundary', s=0.5)
# Add red box for the domain boundary
box_x = [0, side_length, side_length, 0, 0]
box_y = [0, 0, side_length, side_length, 0]
plt.plot(box_x, box_y, color='black', label='Domain Boundary')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
# %%
