from .lenet import *
from .alexnet import *
from .vgg import *
from .vggResidual import *
from .vggResidualProposed import *
from .preresnet import *
from .preresneXt import *
from .densenet import *
from  .sePreResNeXt import *
from  .Bam_PreResNeXt import *
from  .GE_PreResNeXt import *
from .SK_PreResNeXt import *


def get_model(config):
    return globals()[config.architecture](config.num_classes)
