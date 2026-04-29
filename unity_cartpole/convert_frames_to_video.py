import argparse
from pathlib import Path
import imageio.v2 as imageio


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames_dir', default='ReplayFrames')
    p.add_argument('--out_mp4', default='replay.mp4')
    p.add_argument('--out_gif', default='replay.gif')
    p.add_argument('--fps', type=int, default=30)
    args = p.parse_args()

    frames_dir = Path(args.frames_dir)
    files = sorted(frames_dir.glob('frame_*.png'))
    if not files:
        raise SystemExit(f'No frames found in {frames_dir}')

    imgs = [imageio.imread(f) for f in files]
    imageio.mimsave(args.out_mp4, imgs, fps=args.fps, codec='libx264', quality=8)
    imageio.mimsave(args.out_gif, imgs, fps=min(args.fps, 20))
    print(f'Wrote {args.out_mp4} and {args.out_gif} from {len(files)} frames')


if __name__ == '__main__':
    main()
