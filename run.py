from tools import pipeline
import parameters
import glob
import matplotlib.pyplot as plt


def main():
    # Main function that executes the functions desired above

    image_path = parameters.path()
    png_hist = '/Users/aidan/Desktop/aidan_summer/Week_Tasks/Week_9/tma-extracted/tma_extracted_png'
    test_hist = '/Users/aidan/Desktop/aidan_summer/Week_Tasks/Week_9/test'  #tma-extracted/tma_extracted_png'  # Path with image files (png)
    path = '/Users/aidan/Desktop/aidan_summer/Week_Tasks/Week_9/save_images/segment1/' # Path to save CSV file
    path = parameters.save()
    ### Uncomment this to run - Raquel
    #test_hist = '/Users/engs1348/Raquel/Nottingham-TMAs/tma-extracted'
    #png_hist = '/Users/engs1348/Raquel/Nottingham-TMAs/tma-extracted'
    #path = '/Users/engs1348/Raquel/githubRepositoryWorkingFiles/Histology_Aidan'

    img_set = image_path  # Image set that is to be analyzed
    img_files = glob.glob(img_set + '/*.png')

    output_name = []
    output_nest = []
    output_area = []
    output_perimeter = []
    output_eccentricity = []
    output_filled_area = []
    output_roundness = []
    output_circularity = []

    out_avg_area = []
    out_avg_perim = []
    out_avg_eccen = []
    out_avg_filled = []
    out_avg_roundness = []
    out_avg_circularity = []

    out_tot_area = []
    out_tot_perim = []

    std_dev_area = []
    std_dev_perimeter = []
    std_dev_eccentricity = []
    std_dev_filled_area = []
    std_dev_roundness = []
    std_dev_circularity = []

    for im in img_files:
        pipeline.display_image(im)
        # pipeline.save_image(save_path=path, img=im)
        # nest, area, perimeter, eccentricity, filled_area, avg_area, avg_perim, avg_eccen, avg_filled, roundness,\
        # circularity, avg_roundness, avg_circularity, tot_area, tot_perim, std_area, std_perimeter, std_eccentricity,\
        # std_filled_area, std_roundness, std_circularity, name = pipeline.get_data(im)
        #
        # output_name.append(name)
        # output_nest.append(nest)
        # output_area.append(area)
        # output_perimeter.append(perimeter)
        # output_eccentricity.append(eccentricity)
        # output_filled_area.append(filled_area)
        # output_roundness.append(roundness)
        # output_circularity.append(circularity)
        # out_avg_area.append(avg_area)
        # out_avg_perim.append(avg_perim)
        # out_avg_eccen.append(avg_eccen)
        # out_avg_filled.append(avg_filled)
        # out_avg_roundness.append(avg_roundness)
        # out_avg_circularity.append(avg_circularity)
        #
        # out_tot_area.append(tot_area)
        # out_tot_perim.append(tot_perim)
        #
        # std_dev_area.append(std_area)
        # std_dev_perimeter.append(std_perimeter)
        # std_dev_eccentricity.append(std_eccentricity)
        # std_dev_filled_area.append(std_filled_area)
        # std_dev_roundness.append(std_roundness)
        # std_dev_circularity.append(std_circularity)

    output_data = [output_name,
                   output_nest,
                   output_area,
                   std_dev_area,
                   out_tot_area,
                   out_avg_area,
                   output_perimeter,
                   std_dev_perimeter,
                   out_tot_perim,
                   out_avg_perim,
                   output_circularity,
                   std_dev_circularity,
                   out_avg_circularity,
                   output_roundness,
                   std_dev_roundness,
                   out_avg_roundness,
                   output_eccentricity,
                   std_dev_eccentricity,
                   out_avg_eccen,
                   output_filled_area,
                   std_dev_filled_area,
                   out_avg_filled]

    # print output_data

    # pipeline.write_csv(output_data, save_path='/Users/aidan/Desktop/aidan_summer/Week_Tasks/Week_9')

    ##output_path = '/Users/engs1348/Raquel/githubRepositoryWorkingFiles/Histology_Aidan'
    #write_csv(output_data, save_path=output_path)

    plt.show()

if __name__ == "__main__":
    main()
