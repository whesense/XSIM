import enum


class BoxType(enum.Enum):
  UNKNOWN = 0
  VEHICLE = 1
  PEDESTRIAN = 2
  SIGN = 3
  CYCLIST = 4


class CameraName(enum.Enum):
  UNKNOWN = 0
  FRONT = 1
  FRONT_LEFT = 2
  FRONT_RIGHT = 3
  SIDE_LEFT = 4
  SIDE_RIGHT = 5


class RollingShutterReadOutDirection(enum.Enum):
  UNKNOWN = 0
  TOP_TO_BOTTOM = 1
  LEFT_TO_RIGHT = 2
  BOTTOM_TO_TOP = 3
  RIGHT_TO_LEFT = 4
  GLOBAL_SHUTTER = 5
