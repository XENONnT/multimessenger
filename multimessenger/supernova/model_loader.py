
import os
import numpy as np


def _load_from_X(*args):
    """ As long as it returns
        args = (time, mean_E, Nve, Nave, Nvx, Lve, Lave, Lvx)
    """
    # args = somestuff()
    raise NotImplementedError


def _load_light_curve(mass, Z, t_revival, path=None):
    """ The data are taken from http://asphwww.ph.noda.tus.ac.jp/snn/
    Load data from file;
    File structure is
    `    time_step,\n
         col1 col2 col3 col4 col5 col6 \n
    20x  col1 col2 col3 col4 col5 col6 \n
         col1 col2 col3 col4 col5 col6 \n
    `
    This function is way slower than Ricardo Peres's, but more readable
    """
    available_mass = [13, 20, 30, 50]
    available_Z = [0.02, 0.004]
    available_t_revival = [100, 200, 300]
    assert mass in available_mass, "Required mass not in DB."
    assert Z in available_Z, "Required metallicity not in DB."
    assert t_revival in available_t_revival, "Required revival time not in DB."

    B = 0
    if Z == 0.004:
        B = 1
    C = int(t_revival / 100)

    default_path = f'/dali/lgrandi/peres/SN/Light_Curve_DB/intpdata/'
    dataname = f"intp{mass}{B}{C}.data"
    if path is not None:
        filepath = os.path.join(path ,dataname)
    else:
        filepath = os.path.join(default_path, dataname)
    _f = open(filepath, 'r')
    f = _f.readlines()
    _f.close()
    # The 0th line is the time
    # following 20 lines are the data
    # there is an empty line and a line for next time step after every 20 lines
    skiprows = [0]
    for i in range(int(len(f) / 22)):
        skiprows.append(21 + i * 22)
        skiprows.append(22 + i * 22)

    # make a file without the empty lines or time lines
    f_filt = [fval for fval in f if f.index(fval) not in skiprows]
    f_new = open('tmp', 'w+')
    f_new.writelines(f_filt)
    f_new.close()
    # remove tmp if you like

    time = np.array([float(fval.split('\n')[0]) for fval in f if f.index(fval) in skiprows and fval != ' \n'])
    E_bins_left, E_bins_right, N_ve, N_ave, N_vx, L_ve, L_ave, L_vx = np.loadtxt('tmp', unpack=True)
    row_nr = int(len(L_vx) / 20)
    # the energies are same at a given time index
    # self.t = time  # *u.s
    E_bins_left = E_bins_left.reshape(row_nr, 20)[0, :]  # * u.MeV
    E_bins_right = E_bins_right.reshape(row_nr, 20)[0, :]  # * u.MeV
    mean_E = (E_bins_right + E_bins_left) / 2

    Nve = N_ve.reshape(row_nr, 20)  # * (u.count*u.MeV**-1 * u.s**-1)
    Nave = N_ave.reshape(row_nr, 20)  # * (u.count*u.MeV**-1 * u.s**-1)
    Nvx = 4 * N_vx.reshape(row_nr, 20)  # * (u.count*u.MeV**-1 * u.s**-1)
    Lve = L_ve.reshape(row_nr, 20)  # * (u.erg * u.MeV**-1 * u.s**-1)
    Lave = L_ave.reshape(row_nr, 20)  # * (u.erg * u.MeV**-1 * u.s**-1)
    Lvx = L_vx.reshape(row_nr, 20)  # * (u.erg * u.MeV**-1 * u.s**-1)
    return time, mean_E, Nve, Nave, Nvx, Lve, Lave, Lvx






    # def _load_light_curve(self, path=None):
    #     """ The data are taken from http://asphwww.ph.noda.tus.ac.jp/snn/
    #     Load data from file;
    #     File structure is
    #     `    time_step,\n
    #          col1 col2 col3 col4 col5 col6 \n
    #     20x  col1 col2 col3 col4 col5 col6 \n
    #          col1 col2 col3 col4 col5 col6 \n
    #     `
    #     This function is way slower than Ricardo Peres's, but more readable
    #     """
    #     mass = self.M
    #     Z = self.Z
    #     t_revival = self.t_revival
    #     available_mass = [13, 20, 30, 50]
    #     available_Z = [0.02, 0.004]
    #     available_t_revival = [100, 200, 300]
    #     assert mass in available_mass, "Required mass not in DB."
    #     assert Z in available_Z, "Required metallicity not in DB."
    #     assert t_revival in available_t_revival, "Required revival time not in DB."
    #
    #     B = 0
    #     if Z == 0.004:
    #         B = 1
    #     C = int(t_revival / 100)
    #
    #     default_path = f'/dali/lgrandi/peres/SN/Light_Curve_DB/intpdata/'
    #     dataname = f"intp{mass}{B}{C}.data"
    #     if path is not None:
    #         filepath = os.path.join(path,dataname)
    #     else:
    #         filepath = os.path.join(default_path, dataname)
    #     _f = open(filepath, 'r')
    #     f = _f.readlines()
    #     _f.close()
    #     # The 0th line is the time
    #     # following 20 lines are the data
    #     # there is an empty line and a line for next time step after every 20 lines
    #     skiprows = [0]
    #     for i in range(int(len(f) / 22)):
    #         skiprows.append(21 + i * 22)
    #         skiprows.append(22 + i * 22)
    #
    #     # make a file without the empty lines or time lines
    #     f_filt = [fval for fval in f if f.index(fval) not in skiprows]
    #     f_new = open('tmp', 'w+')
    #     f_new.writelines(f_filt)
    #     f_new.close()
    #     # remove tmp if you like
    #
    #     time = np.array([float(fval.split('\n')[0]) for fval in f if f.index(fval) in skiprows and fval != ' \n'])
    #     E_bins_left, E_bins_right, N_ve, N_ave, N_vx, L_ve, L_ave, L_vx = np.loadtxt('tmp', unpack=True)
    #     row_nr = int(len(L_vx) / 20)
    #     # the energies are same at a given time index
    #     # self.t = time  # *u.s
    #     E_bins_left = E_bins_left.reshape(row_nr, 20)[0, :]  # * u.MeV
    #     E_bins_right = E_bins_right.reshape(row_nr, 20)[0, :]  # * u.MeV
    #     mean_E = (E_bins_right + E_bins_left) / 2
    #
    #     Nve = N_ve.reshape(row_nr, 20)  # * (u.count*u.MeV**-1 * u.s**-1)
    #     Nave = N_ave.reshape(row_nr, 20)  # * (u.count*u.MeV**-1 * u.s**-1)
    #     Nvx = 4 * N_vx.reshape(row_nr, 20)  # * (u.count*u.MeV**-1 * u.s**-1)
    #     Lve = L_ve.reshape(row_nr, 20)  # * (u.erg * u.MeV**-1 * u.s**-1)
    #     Lave = L_ave.reshape(row_nr, 20)  # * (u.erg * u.MeV**-1 * u.s**-1)
    #     Lvx = L_vx.reshape(row_nr, 20)  # * (u.erg * u.MeV**-1 * u.s**-1)
    #     return time, mean_E, Nve, Nave, Nvx, Lve, Lave, Lvx

