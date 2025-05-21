import numpy as np
import csv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from pyransac3d import Plane
from mpl_toolkits.mplot3d import Axes3D

def generate_horizontal_plane(width, length, density=1000):
    x = np.random.uniform(0, width, density)
    y = np.random.uniform(0, length, density)
    z = np.zeros(density)
    return np.column_stack((x, y, z))

def generate_vertical_plane(width, height, density=1000):
    x = np.random.uniform(0, width, density)
    y = np.zeros(density)
    z = np.random.uniform(0, height, density)
    return np.column_stack((x, y, z))

def generate_cylindrical_surface(radius, height, density=1000):
    theta = np.random.uniform(0, 2 * np.pi, density)
    z = np.random.uniform(0, height, density)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack((x, y, z))

def fit_plane_ransac(points, threshold=0.01, max_iterations=1000):
    best_plane = None
    best_inliers_count = 0
    best_inliers = None

    n_points = points.shape[0]

    for _ in range(max_iterations):
        sample_indices = np.random.choice(n_points, 3, replace=False)
        sample_points = points[sample_indices]

        v1 = sample_points[1] - sample_points[0]
        v2 = sample_points[2] - sample_points[0]
        normal = np.cross(v1, v2)
        if np.linalg.norm(normal) == 0:
            continue 
        normal = normal / np.linalg.norm(normal)
        D = -np.dot(normal, sample_points[0])

        distances = np.abs(np.dot(points, normal) + D)

        inliers = distances < threshold
        inliers_count = np.sum(inliers)

        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_plane = (normal, D)
            best_inliers = inliers

    return best_plane, best_inliers

def analyze_plane_file(filename, threshold=0.02):
    pts = np.loadtxt(filename)
    (n, D), inliers = fit_plane_ransac(pts)
    avg = np.mean(np.abs(pts @ n + D))

    if avg > threshold:
        orient = "nie jest płaszczyzną"
    else:
        zc = abs(n[2])
        orient = "pozioma" if zc > 0.9 else "pionowa" if zc < 0.1 else "ukośna"

    print(
        f"{filename} -> "
        f"{n[0]:.3f}x+{n[1]:.3f}y+{n[2]:.3f}z+{D:.3f}=0 | "
        f"inliers {inliers.sum()}/{len(pts)} | "
        f"avg_dist {avg:.4f} | {orient}"
    )

analyze_plane_file('horizontal_surface.xyz')
analyze_plane_file('vertical_surface.xyz')
analyze_plane_file('cylindrical_surface.xyz')

np.savetxt("horizontal_surface.xyz", generate_horizontal_plane(10, 10, 1000), fmt="%.6f", delimiter=' ')
np.savetxt("vertical_surface.xyz", generate_vertical_plane(10, 10, 1000), fmt="%.6f", delimiter=' ')
np.savetxt("cylindrical_surface.xyz", generate_cylindrical_surface(5, 10, 1000), fmt="%.6f", delimiter=' ')
all_points = np.vstack([
    generate_horizontal_plane(10, 10, 1000),
    generate_vertical_plane(10, 10, 1000),
    generate_cylindrical_surface(5, 10, 1000)
])
np.savetxt("all_surfaces.xyz", all_points, fmt="%.6f", delimiter=' ')

def load_xyz_csv(filename):
    points = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', skipinitialspace=True)
        for row in reader:
            if len(row) < 3:
                continue
            x, y, z = map(float, row[:3])
            points.append((x, y, z))
    return np.array(points)

def cluster_xyz_file_csv(filenames, k=3, save_clusters=False):
    if isinstance(filenames, str):
        filenames = [filenames]

    pts = np.vstack([load_xyz_csv(fn) for fn in filenames])

    kmeans = KMeans(n_clusters=k, random_state=42).fit(pts)
    labels = kmeans.labels_

    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"Klaster {u}: {c} punktów")

    if save_clusters:
        for i in range(k):
            cluster_pts = pts[labels == i]
            np.savetxt(f"cluster_{i}.xyz", cluster_pts, fmt="%.6f", delimiter=' ')

    print("Współrzędne centrów klastrów:\n", kmeans.cluster_centers_)
    return labels, kmeans.cluster_centers_

labels, centers = cluster_xyz_file_csv("all_surfaces.xyz", k=3, save_clusters=True)

def cluster_dbscan(filename, eps=0.1, min_samples=10):
    pts = load_xyz_csv(filename)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    labels = db.labels_
    unique, counts = np.unique(labels, return_counts=True)
    print(f"DBSCAN na {filename}:")
    for lbl, cnt in zip(unique, counts):
        tag = "szum" if lbl == -1 else f"klaster {lbl}"
        print(f"  {tag}: {cnt} punktów")
    return pts, labels

def fit_plane_pyransac(filename, thresh=0.02, max_iter=1000):
    pts = load_xyz_csv(filename)
    plane = Plane()
    best_eq, inliers = plane.fit(pts, thresh=thresh, maxIteration=max_iter)
    a, b, c, d = best_eq
    normal = np.array([a, b, c])
    avg_dist = np.mean(np.abs(pts @ normal + d))
    if avg_dist > thresh:
        orient = "nie jest płaszczyzną"
    else:
        zc = abs(normal[2])
        orient = "pozioma" if zc > 0.9 else "pionowa" if zc < 0.1 else "ukośna"
    print(f"\nRANSAC3D na {filename}:")
    print(f"  równanie: {a:.3f}x+{b:.3f}y+{c:.3f}z+{d:.3f}=0")
    print(f"  inliers: {len(inliers)}/{len(pts)}")
    print(f"  avg_dist: {avg_dist:.4f}")
    print(f"  wynik: {orient}")


pts_all, labels_all = cluster_dbscan('all_surfaces.xyz', eps=2, min_samples=20)

files = ['horizontal_surface.xyz',
         'vertical_surface.xyz',
         'cylindrical_surface.xyz']

for fn in files:
    fit_plane_pyransac(fn, thresh=0.02, max_iter=1000)