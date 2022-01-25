import os
import sys
import random
import argparse
import imageio
import numpy as np
import trimesh

import torch
import neural_renderer as nr


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mesh_file", type=str, required=True, help="mesh file to turn into a gif"
    )
    parser.add_argument("--out_dir", type=str, required=True, help="output directory")
    parser.add_argument(
        "--azim", type=float, default=0.0, help="start value of azimuth"
    )
    parser.add_argument("--elev", type=float, default=30.0, help="elevation")
    parser.add_argument(
        "--cam_dist", type=float, default=1.25, help="distance of camera to object"
    )
    parser.add_argument(
        "--img_size", type=int, default=512, help="size of rendered images"
    )
    parser.add_argument(
        "--n_views", type=int, default=50, help="number of views to render"
    )
    parser.add_argument("--textured", action="store_true", help="load a textured mesh")

    return parser.parse_args()


def get_projection(
    az, el, distance, focal_length=35, img_w=256, img_h=256, sensor_size_mm=32.0
):
    """Calculate 4x3 3D to 2D projection matrix given viewpoint parameters."""

    # Calculate intrinsic matrix.
    f_u = focal_length * img_w / sensor_size_mm
    f_v = focal_length * img_h / sensor_size_mm
    u_0 = img_w / 2
    v_0 = img_h / 2
    K = np.matrix(((f_u, 0, u_0), (0, f_v, v_0), (0, 0, 1)))

    # Calculate rotation and translation matrices.
    sa = np.sin(np.radians(az))
    ca = np.cos(np.radians(az))
    R_azimuth = np.transpose(np.matrix(((ca, 0, sa), (0, 1, 0), (-sa, 0, ca))))
    se = np.sin(np.radians(el))
    ce = np.cos(np.radians(el))
    R_elevation = np.transpose(np.matrix(((1, 0, 0), (0, ce, -se), (0, se, ce))))

    # fix up camera
    se = np.sin(np.radians(180))
    ce = np.cos(np.radians(180))
    R_cam = np.transpose(np.matrix(((ce, -se, 0), (se, ce, 0), (0, 0, 1))))
    T_world2cam = np.transpose(np.matrix((0, 0, distance)))
    RT = np.hstack((R_cam @ R_elevation @ R_azimuth, T_world2cam))
    return K, RT


def main():
    args = parse_args()

    try:
        if args.textured:
            print("Textured meshes not supported yet.")
            sys.exit()
        else:
            if args.mesh_file.endswith(".obj"):
                mesh = trimesh.load(args.mesh_file)
            else:
                mesh = trimesh.load(args.mesh_file)
    except RuntimeError:
        print("Not a valid mesh file to load.")
        sys.exit()

    images = []
    verts = torch.tensor(mesh.vertices[None, ...], dtype=torch.float32).to(device)
    faces = torch.tensor(mesh.faces[None, ...], dtype=torch.float32).to(device)
    textures = torch.tensor(np.ones((1, faces.shape[1], 2, 2, 2, 3), "float32")).to(
        device
    )

    for ii in range(args.n_views):
        azimuth = args.azim - ii * (360 / args.n_views)
        intrinsic, extrinsic = get_projection(
            azimuth, args.elev, args.cam_dist, img_w=args.img_size, img_h=args.img_size
        )

        # set up renderer
        K_cuda = (
            torch.tensor(intrinsic[np.newaxis, :, :].copy())
            .float()
            .to(device)
            .unsqueeze(0)
        )
        R_cuda = (
            torch.tensor(extrinsic[np.newaxis, 0:3, 0:3].copy())
            .float()
            .to(device)
            .unsqueeze(0)
        )
        t_cuda = (
            torch.tensor(extrinsic[np.newaxis, np.newaxis, 0:3, 3].copy())
            .float()
            .to(device)
            .unsqueeze(0)
        )
        renderer = nr.Renderer(
            image_size=args.img_size,
            orig_size=args.img_size,
            K=K_cuda,
            R=R_cuda,
            t=t_cuda,
            anti_aliasing=False,
            light_direction=[0, -1, 0],
        )

        img_out, depth_out, silhouette_out = renderer(verts, faces, textures)
        images.append(img_out[0].permute(1, 2, 0))

    images = np.stack([img.detach().cpu().numpy() for img in images])

    # Save the gif
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    images = (255 * images).astype(np.uint8)
    gif_filename = os.path.splitext(os.path.basename(args.mesh_file))
    gif_path = os.path.join(args.out_dir, gif_filename[0] + ".gif")
    imageio.mimsave(gif_path, images[..., :3])


if __name__ == "__main__":
    main()
