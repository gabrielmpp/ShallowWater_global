import xarray as xr
import numpy as np
from maisterlib.config_oper import DATAPATH
import matplotlib.pyplot as plt
from toolbox.physical_constants import EARTH_RADIUS
from numba import jit


@jit(nopython=True)
def fourth_order_derivative(arr: np.ndarray, dim=0):
    """
    2D numpy array with dims [lat, lon]
    :param arr:
    :return:
    """
    # assert isinstance(arr, np.ndarray), 'Input must be numpy array'
    output = np.zeros_like(arr)

    if dim == 0:
        ysize = np.shape(arr)[0]
        for lat_idx in range(2, np.shape(arr - 2)[0]):
            for lon_idx in range(np.shape(arr)[1]):
                output[lat_idx, lon_idx] = (4 / 3) * (arr[(lat_idx + 1), lon_idx] -
                                                      arr[(lat_idx - 1), lon_idx]) / 2 \
                                           - (1 / 3) * (arr[(lat_idx + 2), lon_idx] -
                                                        arr[(lat_idx - 2), lon_idx]) / 4

        #  First order uncentered derivative for points close to the poles
        for lat_idx in [0, 1]:
            for lon_idx in range(np.shape(arr)[1]):
                output[lat_idx, lon_idx] = (arr[(lat_idx + 1), lon_idx] -
                                            arr[lat_idx, lon_idx]) / 2
        for lat_idx in [-1, -2]:
            for lon_idx in range(np.shape(arr)[1]):
                output[lat_idx, lon_idx] = (arr[lat_idx, lon_idx] -
                                            arr[lat_idx - 1, lon_idx]) / 2
    elif dim == 1:
        xsize = np.shape(arr)[1]
        for lat_idx in range(np.shape(arr)[0]):
            for lon_idx in range(np.shape(arr)[1]):
                output[lat_idx, lon_idx] = (4 / 3) * (arr[lat_idx, (lon_idx + 1) % xsize] -
                                                      arr[lat_idx, (lon_idx - 1) % xsize]) / 2 \
                                           - (1 / 3) * (arr[lat_idx, (lon_idx + 2) % xsize] -
                                                        arr[lat_idx, (lon_idx - 2) % xsize]) / 4

    return output


def derivative_spherical_coords(da, dim=0):
    EARTH_RADIUS = 6371000  # m
    da = da.sortby('latitude')
    da = da.sortby('longitude')
    da = da.transpose('latitude', 'longitude')
    x = da.longitude.copy() * np.pi / 180
    y = da.latitude.copy() * np.pi / 180
    dx = (np.pi/180) * (da.longitude.values[1] - da.longitude.values[0]) * EARTH_RADIUS * np.cos(y)
    dy = (np.pi/180) * (da.latitude.values[1] - da.latitude.values[0]) * EARTH_RADIUS
    deriv = fourth_order_derivative(da.values, dim=dim)
    deriv = da.copy(data=deriv)

    if dim == 0:
        deriv = deriv / dy
    elif dim == 1:
        deriv = deriv / dx
    else:
        raise ValueError('Dim must be either 0 or 1.')
    return da.copy(data=deriv)


def staggerArakawaC(da_u, da_v, da_h):
    da_u = da_u.copy()
    da_v = da_v.copy()
    da_h = da_h.copy()

    d_lat = da_h.latitude.diff('latitude').values[0]
    d_lon = da_h.longitude.diff('longitude').values[0]
    new_lon_u = np.concatenate([da_u.longitude.values[0].reshape(1,) - d_lon/2,
                                (da_u.longitude.values + d_lon/2) % 360])
    new_lat_v = np.concatenate([da_v.latitude.values[0].reshape(1,) - d_lat/2, da_v.latitude.values + d_lat/2])

    da_v = da_v.reindex(latitude=new_lat_v, fill_value=0)
    da_u = da_u.reindex(longitude=new_lon_u, fill_value=0)
    return da_u, da_v, da_h


def gaussian_initial_condition(nx, ny, sigma=.5, mu=0):
    f = lambda x: np.exp(-((x - mu) ** 2 / (2.0 * sigma ** 2)))
    x, y = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
    d = np.sqrt(x * x + y * y)
    return .1 * f(d)

def FTCS(da_u, da_v, da_h, int_times, dt, H, gamma):
    u_outs = []
    v_outs = []
    h_outs = []
    coriolis = 2 * omega * np.sin(da_h.latitude * np.pi/180)
    for int_time in int_times:
        print(str(int(int_time * 100/int_times[-1])) + '%')
        da_h = da_h - dt * H * (derivative_spherical_coords(da_u, dim=1) +
                                derivative_spherical_coords(da_v, dim=0))
        if int_time % 2 == 0:
            da_u = da_u + dt * (coriolis * da_v) - g * dt * derivative_spherical_coords(da_h, dim=1) - da_u * gamma
            da_v = da_v - dt * (coriolis * da_u) - g * dt * derivative_spherical_coords(da_h, dim=0) -da_v * gamma
            da_v = da_v.where(np.abs(da_v.latitude) < 86, 0)
        else:
            da_v = da_v - dt * (coriolis * da_u) - g * dt * derivative_spherical_coords(da_h, dim=0) - da_v * gamma
            da_v = da_v.where(np.abs(da_v.latitude) < 86, 0)
            da_u = da_u + dt * (coriolis * da_v) - g * dt * derivative_spherical_coords(da_h, dim=1) - da_u * gamma

        u_outs.append(da_u.copy().expand_dims('time'))
        v_outs.append(da_v.copy().expand_dims('time'))
        h_outs.append(da_h.copy().expand_dims('time'))
        # if int_time % (1000) == 0:
        #     fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.Robinson()})
        #     da_h.plot(ax=ax, transform=ccrs.PlateCarree(), )
        #     ax.coastlines()
        #     plt.show()

    da_u = xr.concat(u_outs, dim='time')
    da_v = xr.concat(v_outs, dim='time')
    da_h = xr.concat(h_outs, dim='time')
    da_u = da_u.assign_coords(time=int_times)
    da_v = da_v.assign_coords(time=int_times)
    da_h = da_h.assign_coords(time=int_times)
    return da_u, da_v, da_h

def plot(da_h):
    import cartopy.crs as ccrs

    for t in np.arange(0, da_h.time.values.shape[0], 20):
        print(np.int(100*(t/da_h.time.values.shape[0])))
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.Robinson(central_longitude=180)})
        da_h.isel(time=t).plot(ax=ax, transform=ccrs.PlateCarree(), vmin=-0.1,
                               vmax=1, cmap='nipy_spectral', add_colorbar=False)
        ax.coastlines()
        plt.savefig(f'figs/{t:04d}.png', dpi=400,bbox_inches='tight')
        plt.close()

integration_length = 1  # days
mu = 0
sigma = .3

# --- Fixed constants: --- #
gamma = 0
H = 1e4
g = 10
omega = 2 * np.pi / 86400
dt = np.sqrt(g*H)

# --- Grid setup --- #
lats = np.arange(-86, 87, 1)
lons = np.arange(1, 360, 1)
dx = 6371000 * np.cos(np.max(lats)*np.pi/180) * (lats[1] - lats[0])
c = np.sqrt(g*H)  # speed of gravity waves
dt_max = .1*dx/(np.sqrt(2)*c)
# dt = 50  # Time interval in seconds. Must be smaller than dt_max
times_in_seconds = np.arange(0, integration_length * 86400, dt_max)
da_h = xr.DataArray(10*gaussian_initial_condition(lons.shape[0], lats.shape[0], sigma=.3, mu=0), dims=('latitude', 'longitude'),
                    coords={'longitude': lons, 'latitude': lats})
da_u = xr.zeros_like(da_h)
da_v = xr.zeros_like(da_h)

# --- Running integration --- #
da_u, da_v, da_h = FTCS(da_u, da_v, da_h, times_in_seconds, dt, H, gamma)
# --- Storing outputs --- #
da_u.to_netcdf('u_out.nc')
da_v.to_netcdf('v_out.nc')
da_h.to_netcdf('h_out.nc')



