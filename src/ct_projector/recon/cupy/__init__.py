BACKEND = 'cupy'

from .sqs_algorithms import sqs_gaussian_one_step, nesterov_acceleration, sqs_one_step  # noqa

from .sqs_motion_version import sqs_gaussian_one_step_motion, nesterov_acceleration_motion  # noqa