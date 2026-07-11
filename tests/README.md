# Offline tests

Run the hardware-independent suite from the repository root:

```powershell
python -m pytest
```

`pytest.ini` limits default discovery to `tests/`. Legacy ROS, camera, and
gripper probes remain under `test/` and are intentionally excluded because
some of them connect to hardware during module import.

The offline suite uses recording fake backends and temporary HDF5 files. It
does not require a robot, ROS graph, camera, or gripper.

Qt component tests run with `QT_QPA_PLATFORM=offscreen`, so they create real
widgets and process Qt events without opening desktop windows.

Install only the packages needed by this suite with:

```powershell
python -m pip install -r requirements-test.txt
```
