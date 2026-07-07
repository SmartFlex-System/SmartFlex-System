
## Input Columns

Each time-series CSV should contain:

```text
aX,aY,aZ,Gx,Gy,Gz,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16
```

where:

- `aX, aY, aZ`: accelerometer channels.
- `Gx, Gy, Gz`: gyroscope channels.
- `F1` to `F16`: plantar pressure sensor channels.

## Target Columns

For supervised training or evaluation, also include:

```text
Fx,Fy,Fz,Mx,My,Mz,COPx,COPy
```

These represent three-dimensional ground reaction forces, three-dimensional moments, and center-of-pressure coordinates.


