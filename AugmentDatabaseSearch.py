import os
from PIL import Image
def process_images(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            # Flip image
            basename = filename.split('.')[0]
            flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_image.save(os.path.join(output_folder, f'{basename}_Augment_flipped.jpg'))
            # Rotate image 90, 180, 270 degrees
            for angle in [90, 180, 270]:
                rotated_image = image.rotate(angle, expand=True)
                rotated_image.save(os.path.join(output_folder, f'{basename}_Augment_rotated_{angle}.jpg'))
                

# Example usage
list_of_folders = ['CLKU32-178RBH-X2359A','EV-108MSO(2091)','CAC4-A-50-50','MLGPM50-100-B','MDUF50-100-P4-DWSE-X1518','CAC4-A-63','CLKU32-178DDH-X2359A','LR-ZB100P']  # Add your folder paths here
for folder in list_of_folders:
    pathfolder = os.path.join('/media/anlab/data-2tb/ANLAB_THUY/ToyotaAR/Dataset/NewData/Database_CROP',folder)
    process_images(pathfolder, '/media/anlab/data-2tb/ANLAB_THUY/ToyotaAR/Dataset/NewData/Database_CROP/AugmentData/'+folder+'_Augment')  # Change to your output folder path
