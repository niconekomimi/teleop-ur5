import pytest

from teleop_control_py.core.resource_manager import (
    CameraRuntimeContext,
    HardwareConflictError,
    ResourceManager,
)


def test_camera_source_alias_is_normalized() -> None:
    manager = ResourceManager()

    assert manager.normalize_camera_source("RS") == "realsense"
    assert manager.normalize_camera_source(" oakd ") == "oakd"


def test_active_driver_blocks_matching_sdk_camera_request() -> None:
    manager = ResourceManager()
    context = CameraRuntimeContext(
        active_camera_drivers=("realsense",),
        active_camera_driver_devices=(("realsense", "ABC123"),),
    )

    with pytest.raises(HardwareConflictError):
        manager.check_camera_availability(
            requester="collector",
            requested_sources=["realsense"],
            requested_serial_numbers=["ABC123"],
            context=context,
        )


def test_different_known_serial_is_available() -> None:
    manager = ResourceManager()
    context = CameraRuntimeContext(
        active_camera_driver_devices=(("realsense", "ABC123"),),
    )

    manager.check_camera_availability(
        requester="inference",
        requested_sources=["realsense"],
        requested_serial_numbers=["XYZ987"],
        context=context,
    )


def test_distinct_views_reject_duplicate_serials() -> None:
    manager = ResourceManager()

    with pytest.raises(HardwareConflictError):
        manager.check_camera_availability(
            requester="inference",
            requested_sources=["oakd", "oakd"],
            requested_serial_numbers=["MXID", "MXID"],
            context=CameraRuntimeContext(),
            require_distinct_views=True,
        )

