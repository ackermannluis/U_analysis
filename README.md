# U_analysis
Universal data analysis tools for atmospheric sciences

Script written in python 3. This file defines multiple functions that can be used for data analysis, with an emphasis in atmospheric sciences.
To use, just place it in the same directory as your code is and import it as any other module (import U_Analysis_main). For full functionality add the other files to the same directory). 

Working in adding it to the python repository such that it can be imported using pip or conda...

This module has hundreds of functions (fairly well organized by theme) some specially useful functions are:  

nc_show_variable_info (which shows all variables inside, their shape, and units, it can also give more info for specific variable)

p_plot(for general (1D arrays) time series or scatter plots, basically a general wrapper for matplotlib, can be used to plot over map if topographical files are present)

p_plot_arr(for general plotting of 2D arrays, where the horizontal and vertical arrays are also provided, basically a general wrapper for matplotlib, can be used to plot over map if topographical files are present)

p_plot_SkewT_sonde(for creating nice skewT plots, needs the skewT_module given below)

wrf_var_search(give it the wrf filename (or nc object) and a keyword and it will print all variables with the keyword in either the name or the description)

create_virtual_sonde_from_wrf(give it radiosonde data and a list of wrf output files and it will create a virtual sonde from wrf data that follows as closely as possible the real radiosonde (accounts for time passage and horizontal displacement)

calculate_mountain_height_from_era5(as the name implies)

calculate_mountain_height_from_WRF(as the name implies)

download_HIM8_2000m(give it the time stamp and the channel number and it gives you the array)

get_himawari8_2000m_NCI(same as above but it is meant to run on gadi, and just gets the desired array from the data for some time and channel)

era5_download_save(downloads era5 data from copernicus data store, just give it times you want and the variables you want and it will do the rest. it requires you to have an account and cdsapi installed)

hysplit_load_freq_endpoints(reads hysplit text files and converts to data arrays)

plot_hysplit_traj(plots hysplit trajectories from data arrays)

download_MSLP(give it the time and it downloads the MSLP chart from the BoM)

merge_multiple_netCDF_by_time_dimension(really useful, as the name implies)

netCDF_crop_timewise(when you don't need the whole file, creates a copy with only the desired time range in it)

plot_arr_over_map_nc(to create quick maps, like panoply, requires basemap)

