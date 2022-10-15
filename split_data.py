import argparse
import os
import shutil


def src_images(dir_path):
    count = 0
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
    return count, os.listdir(dir_path)

def move_to_dest_dirs(src,dest,ndir,nfiles):
    os.makedirs(dest,exist_ok=True)
    for i in range(ndir):
        p = f"{dest}/{i:03d}"
        print(f"mkdir {p}")
        os.makedirs(p,exist_ok=True)
        lfiles = os.listdir(src)
        for f in lfiles[0:nfiles]:
            shutil.move(f"{src}/{f}",p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, help='indir: folder where img are')
    parser.add_argument('--outdir', type=str, help='outdir: folder where img go')
    parser.add_argument('--number', type=int, default=500, help='number: number of img per subfolder')
    args = parser.parse_args()

    count,files = src_images(args.indir)
    n_dir = int(count / args.number) + 1 
    print(f"{args.indir} : {count} files ({n_dir} dirs)")
    move_to_dest_dirs(args.indir,args.outdir,n_dir,args.number)

if __name__ == '__main__':
    main()
