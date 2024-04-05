import argparse
import numpy as np 
import matplotlib.pyplot as plt 
from procedural_noise_functions import *

def main(args):
    # Access the arguments as attributes of args
    print("Dim:", args.dim)
    print("Explicit:", args.explicit)

    if(args.explicit):
        plt.imshow(generate_explicit(args.dim,20,4,sigmas = [1,1,3,4]))
        plt.imsave("noise/explicit1.png",generate_explicit(args.dim,20,4,sigmas = [1,1,3,4]), cmap = "Greys")

        plt.imshow(generate_explicit(args.dim,20,2,sigmas = [4,4,3,4]))
        plt.imsave("noise/explicit2.png",generate_explicit(256,20,2,sigmas = [4,4,3,4]), cmap = "Greys")

        plt.imshow(generate_explicit(args.dim,2,4,sigmas = [3,1,3,1]))
        plt.imsave("noise/explicit3.png",generate_explicit(args.dim,2,4,sigmas = [3,1,3,1]), cmap = "Greys")

        plt.imshow(generate_explicit(args.dim,40,4,sigmas = [6,1,3,1]))
        plt.imsave("noise/explicit4.png",generate_explicit(args.dim,40,4,sigmas = [6,1,3,1]), cmap = "Greys")

    else:
        plt.imshow(generate_perlin(args.dim,30,3))
        plt.imsave("noise/perlin1.png",generate_perlin(args.dim,30,3),cmap = "Greys")

        plt.imshow(generate_perlin(args.dim,12,3))
        plt.imsave("noise/perlin2.png",generate_perlin(args.dim,12,3),cmap = "Greys")

        plt.imshow(generate_perlin(args.dim,6,3))
        plt.imsave("noise/perlin3.png",generate_perlin(args.dim,6,3),cmap = "Greys")


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Generating noise")

    # Add arguments
    parser.add_argument("dim", type=int, help="dimension of the output")
    parser.add_argument("--explicit", action="store_true", help="If set output is explicit noise, perline otherwise")

    # Parse arguments
    args = parser.parse_args()

    # Call the main function
    main(args)