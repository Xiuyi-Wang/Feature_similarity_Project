from PIL import Image, ImageFont, ImageDraw
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wbplot.utils.plots import check_vrange, check_cmap_plt
import os
from os.path import join



#font_file = '/home/lvboh/code_data_baihan/atlas/Atlas/Playfair_Display/static/PlayfairDisplay-Regular.ttf'
font_file = '/home/xiuyi/Data/Atlas/Playfair_Display/arial/arial.ttf'
titlefont = ImageFont.truetype(font=font_file,size=30)
fontprop = fm.FontProperties(fname=font_file,size=36)

image_path = '/home/lvboh/code_data_baihan/figure/FC_400parcel_distance/images'

def make_transparent(img_file):
    """
    Make each white pixel in an image transparent.

    Parameters
    ----------
    img_file : str
        absolute path to a PNG image file_hctsa

    Returns
    -------
    None

    Notes
    -----
    This function overwrites the existing file_hctsa.

    """

    img = Image.open(img_file)
    img = img.convert("RGBA")
    pixdata = img.load()
    width, height = img.size
    for y in range(height):
        for x in range(width):
            if pixdata[x, y] == (0, 0, 0, 255):  # if black
                pixdata[x, y] = (255, 255, 255, 255)  # set alpha = 255
    img.save(img_file, "PNG")

def add_title(img_file,title,title_position=300):
    """
    Add the title of the image: the contrast

    Parameters
    ----------
    img_file : str absolute path to a PNG image file_hctsa
    title: the title being shown on the figure
    Returns
    -------
    None

    Notes
    -----
    This function overwrites the existing file_hctsa.

    """
    img = Image.open(img_file)
    image_editable = ImageDraw.Draw(img)
    image_editable.text((title_position, 15), title, (0,0,0), font=titlefont)
    img.save(img_file, "PNG")

def add_title_middle(img_file,title):
    """
    Add the title of the image: the contrast

    Parameters
    ----------
    img_file : str absolute path to a PNG image file_hctsa
    title: the title being shown on the figure
    Returns
    -------
    None

    Notes
    -----
    This function overwrites the existing file_hctsa.

    """
    img = Image.open(img_file)
    image_editable = ImageDraw.Draw(img)
    image_editable.text((700, 15), title, (0,0,0), font=titlefont)
    img.save(img_file, "PNG")

def add_title_little(img_file,title):
    """
    Add the title of the image: the contrast

    Parameters
    ----------
    img_file : str absolute path to a PNG image file_hctsa
    title: the title being shown on the figure
    Returns
    -------
    None

    Notes
    -----
    This function overwrites the existing file_hctsa.

    """
    font = ImageFont.truetype(font=font_file, size=30)
    img = Image.open(img_file)
    image_editable = ImageDraw.Draw(img)
    image_editable.text((20, 15), title, (0,0,0), font=titlefont)
    img.save(img_file, "PNG")


# add color bar11
def add_color_bar(img_file,vmin_neg_min, vmin_neg_max,vmin_pos_min,vmin_pos_max,color='psych'):
    """
    add a color bar for the image_file
    :param img_file:
    :param vmin: the minimum value for the color bar
    :param vmax: the maximum value for the color bar
    :return: None
    This function overwrites the existing file_hctsa.
    """
    if color =='psych':
        file_color_bar = '/home/xiuyi/Data/Task_Gradient/results/color_bar_psych_fixed.png'
    elif color == 'spectral':
        file_color_bar = '/home/xiuyi/Data/Task_Gradient/results/color_bar_spectral.png'

    # open color bar
    img_colorbar = Image.open(file_color_bar)

    # open the original figure
    img = Image.open(img_file)

    # add the color bar to the original figure
    img.paste(img_colorbar,(180,570))

    # add the ticks
    tick_height = 585
    vmin_neg_mid = np.round((vmin_neg_min+vmin_neg_max)/2,2)
    vmin_pos_mid = np.round((vmin_pos_min+vmin_pos_max)/2,2)
    image_editable = ImageDraw.Draw(img)
    image_editable.text((180, tick_height), str(vmin_neg_min), (0, 0, 0), font=titlefont)
    image_editable.text((380, tick_height), str(vmin_neg_mid), (0, 0, 0), font=titlefont)
    image_editable.text((550, tick_height), str(vmin_neg_max), (0, 0, 0), font=titlefont)
    image_editable.text((660, tick_height), str(vmin_pos_min), (0, 0, 0), font=titlefont)
    image_editable.text((840, tick_height), str(vmin_pos_mid), (0, 0, 0), font=titlefont)
    image_editable.text((1060,tick_height), str(vmin_pos_max), (0, 0, 0), font=titlefont)

    # save it
    img.save(img_file)


num_decimal = 2
# add color bar11
def add_colorbar2(data, img_file, cmap, orientation='horizontal'):
    """
    add a color bar for the image_file
    :param img_file:
    :param vmin: the minimum value for the color bar
    :param vmax: the maximum value for the color bar
    :return: None
    This function overwrites the existing file_hctsa.
    """
    import uuid
    
    # open the original figure
    img = Image.open(img_file) 
    img_x,img_y = img.size
    
    # draw a colorbar and open it
    colorbar_id = uuid.uuid1().hex
    fig_dir = os.path.dirname(img_file)
    file_colorbar = join(fig_dir, 'colorbar_%s.png'%(colorbar_id))
    # baihan raw value
    # draw_colorbar(data, figsize=(8, 1), file_colorbar=file_colorbar, cmap=cmap, orientation=orientation)
    draw_colorbar(data, figsize=(4, 0.13), file_colorbar=file_colorbar, cmap=cmap, orientation=orientation)

    img_colorbar = Image.open(file_colorbar)
    img_colorbar_x, img_colorbar_y = img_colorbar.size

    # add the color bar to the original figure
    img.paste(img_colorbar,(np.int16(img_x/2-img_colorbar_x/2), np.int16(img_y/2-img_colorbar_y/2)) )
    # img.save(img_file, dpi=(300.0, 300.0))

    # add data:
    # divide it to positive and negative values
    data_neg = data[data < 0]
    data_pos = data[data > 0]
    if data_pos.shape[0] != 0:
        vmin_pos_max = np.round(data_pos.max(), num_decimal)
        vmin_pos_min = np.round(data_pos.min(), num_decimal)
    else:
        vmin_pos_max = 0
        vmin_pos_min = 0
    if data_neg.shape[0] != 0:
        vmin_neg_max = np.round(data_neg.max(), num_decimal)
        vmin_neg_min = np.round(data_neg.min(), num_decimal)
    else:
        vmin_neg_max = 0
        vmin_neg_min = 0


    tick_height = 620
    image_editable = ImageDraw.Draw(img)
    vmin_neg_mid = np.round((vmin_neg_min + vmin_neg_max) / 2, 2)
    vmin_pos_mid = np.round((vmin_pos_min + vmin_pos_max) / 2, 2)
    if data_neg.shape[0] != 0 and  data_pos.shape[0] != 0:
        image_editable.text((180, tick_height), str(vmin_neg_min), (0, 0, 0), font=titlefont)
        image_editable.text((380, tick_height), str(vmin_neg_mid), (0, 0, 0), font=titlefont)
        image_editable.text((550, tick_height), str(vmin_neg_max), (0, 0, 0), font=titlefont)

        image_editable.text((660, tick_height), str(vmin_pos_min), (0, 0, 0), font=titlefont)
        image_editable.text((840, tick_height), str(vmin_pos_mid), (0, 0, 0), font=titlefont)
        image_editable.text((1050,tick_height), str(vmin_pos_max), (0, 0, 0), font=titlefont)

    elif data_neg.shape[0] == 0 and  data_pos.shape[0] != 0:
        image_editable.text((180, tick_height), str(vmin_pos_min), (0, 0, 0), font=titlefont)
        image_editable.text((610, tick_height), str(vmin_pos_mid), (0, 0, 0), font=titlefont)
        image_editable.text((1050, tick_height), str(vmin_pos_max), (0, 0, 0), font=titlefont)

    elif data_neg.shape[0] != 0 and data_pos.shape[0] == 0:
        image_editable.text((180, tick_height), str(vmin_neg_min), (0, 0, 0), font=titlefont)
        image_editable.text((610, tick_height), str(vmin_neg_mid), (0, 0, 0), font=titlefont)
        image_editable.text((1050, tick_height), str(vmin_neg_max), (0, 0, 0), font=titlefont)

    # save it
    img.save(img_file, dpi=(300.0, 300.0))

    os.system('rm -v %s'%(file_colorbar))

def draw_colorbar(data, figsize, file_colorbar, cmap, vrange=None, orientation='horizontal'):
    

    if vrange is None:
        vrange=(np.min(data), np.max(data)) 
    else:
        check_vrange(vrange)
    
    vmin=np.int16(np.floor(vrange[0]))
    vmax=np.round(vrange[1],1)
    if vmin>=0:
        
        vmid=(vmin+vmax)/2
        if vmax <=6:
            vmid=np.round(vmid,2)
        else:
            vmid=np.round(vmid,1)
    else:
        vmid=0

    

    ticks = [vmin, vmid, vmax]
    #draw a custom colorbar and save a temporary picture
    fig = plt.figure(constrained_layout=True,figsize=figsize)
    #plt.rcParams.update({'font.sans-serif':'Arial'})
    ax = fig.subplots(1,1)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    fcb = mpl.colorbar.ColorbarBase(norm=norm, cmap=mpl.cm.get_cmap(cmap), orientation=orientation,ax=ax ,ticks = ticks )
    fcb.ax.set_xticklabels(ticks, fontproperties=fontprop)

    
    #save
    fig.savefig(file_colorbar, dpi=300,transparent=True)
    plt.close()