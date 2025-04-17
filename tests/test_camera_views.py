# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.


from locate3d_data.arkitscenes_dataset import ARKitScenesDataset
from locate3d_data.scannet_dataset import ScanNetDataset
from locate3d_data.scannetpp_dataset import ScanNetPPDataset
from pytest import approx
from pytest import approx as ptapprox
import functools


ARKIT_DIRECTORY = '[arkit_dir']
SCANNET_DIRECTORY = '[scannet_dir]'
SCANNETPP_DIRECTORY = '[scannetpp_dir]'

def test_scannet_camera_views():
    dataset = ScanNetDataset(SCANNET_DIRECTORY)

    # Scene scene0011_00
    data = dataset.get_camera_views('scene0011_00')

    ### RGB
    assert list(data['rgb'].shape) == [80, 3, 480, 640]
    assert data['rgb'].mean().item() == approx(0.375026136636734)
    assert data['rgb'].var().item() == approx(0.06894242018461227)
    
    assert data['rgb'][58:65,1:2,427:463,132:302].mean().item() == approx(0.3139995038509369)
    assert data['rgb'][35:37,0:1,398:412,238:549].var().item() == approx(0.03924134746193886)

    ### Depth
    assert list(data['depth_zbuffer'].shape) == [80, 480, 640]
    assert data['depth_zbuffer'].mean().item() == approx(1.5685722827911377)
    
    assert data['depth_zbuffer'].var().item() == approx(1.4845291376113892)
    assert data['depth_zbuffer'][56:79,357:444,69:408].mean().item() == approx(1.3986636400222778)
    assert data['depth_zbuffer'][53:56,13:318,162:490].var().item() == approx(2.790724754333496)

    ### Cam to world
    assert list(data['cam_to_world'].shape) == [80, 4, 4]
    assert data['cam_to_world'].mean().item() == approx(0.07691331207752228)
    assert data['cam_to_world'].var().item() == approx(0.6674786806106567)
    
    assert data['cam_to_world'][1:30,2:3,1:2].mean().item() == approx(-0.9091897010803223)
    assert data['cam_to_world'][1:64,1:3,0:3].var().item() == approx(0.25012633204460144)

    ### Cam K
    assert list(data['cam_K'].shape) == [80, 3, 3]
    assert data['cam_K'].mean().item() == approx(191.04637145996094)
    
    assert data['cam_K'].var().item() == approx(55736.62109375)
    assert data['cam_K'][54:79,1:2,0:1].mean().item() == approx(0.0)
    assert data['cam_K'][26:79,0:1,0:2].var().item() == approx(84197.0625)

    # Scene scene0578_01
    data = dataset.get_camera_views('scene0578_01')

    ### RGB
    assert list(data['rgb'].shape) == [41, 3, 480, 640]
    assert data['rgb'].mean().item() == approx(0.5427114367485046)
    assert data['rgb'].var().item() == approx(0.07469084113836288)
    
    assert data['rgb'][35:40,1:2,424:448,59:217].mean().item() == approx(0.2861787676811218)
    assert data['rgb'][4:12,0:1,435:455,531:584].var().item() == approx(0.024809129536151886)

    ### Depth
    assert list(data['depth_zbuffer'].shape) == [41, 480, 640]
    assert data['depth_zbuffer'].mean().item() == approx(1.8989434242248535)
    
    assert data['depth_zbuffer'].var().item() == approx(0.8993914723396301)
    assert data['depth_zbuffer'][31:40,202:206,208:364].mean().item() == approx(2.0542759895324707)
    assert data['depth_zbuffer'][6:21,420:444,127:589].var().item() == approx(0.24572327733039856)

    ### Cam to world
    assert list(data['cam_to_world'].shape) == [41, 4, 4]
    assert data['cam_to_world'].mean().item() == approx(0.06278422474861145)
    
    assert data['cam_to_world'].var().item() == approx(0.46016907691955566)
    assert data['cam_to_world'][37:38,0:1,2:3].mean().item() == approx(0.4329930245876312)
    assert data['cam_to_world'][28:40,0:1,2:3].var().item() == approx(0.2543269097805023)

    ### Cam K
    assert list(data['cam_K'].shape) == [41, 3, 3]
    assert data['cam_K'].mean().item() == approx(190.98684692382812)
    
    assert data['cam_K'].var().item() == approx(55953.0859375)
    assert data['cam_K'][11:15,1:2,0:1].mean().item() == approx(0.0)
    assert data['cam_K'][28:34,0:1,1:2].var().item() == approx(0.0)
    

def test_arkit_camera_views():
    dataset = ARKitScenesDataset(ARKIT_DIRECTORY)
    
    # Scene 40753679
    data = dataset.get_camera_views('40753679')
    
    ### RGB
    assert list(data['rgb'].shape) == [113, 3, 192, 256]
    assert data['rgb'].mean().item() == approx(0.5546867847442627)
    assert data['rgb'].var().item() == approx(0.06751064956188202)
    
    assert data['rgb'][79:101,1:2,122:165,22:217].mean().item() == approx(0.46292147040367126)
    assert data['rgb'][43:76,1:2,79:151,101:109].var().item() == approx(0.04884002357721329)

    ### Depth
    assert list(data['depth_zbuffer'].shape) == [113, 192, 256]
    assert data['depth_zbuffer'].mean().item() == approx(1.912764072418213)
    assert data['depth_zbuffer'].var().item() == approx(0.40816500782966614)
    
    assert data['depth_zbuffer'][42:97,165:170,240:243].mean().item() == approx(1.2496352195739746)
    assert data['depth_zbuffer'][23:111,117:178,67:88].var().item() == approx(0.22864381968975067)

    ### Cam to world
    assert list(data['cam_to_world'].shape) == [113, 4, 4]
    assert data['cam_to_world'].mean().item() == approx(0.010665361769497395)
    
    assert data['cam_to_world'].var().item() == approx(0.35834044218063354)
    assert data['cam_to_world'][70:97,0:1,1:3].mean().item() == approx(-0.1123739629983902)
    assert data['cam_to_world'][55:109,0:1,2:3].var().item() == approx(0.42586860060691833)

    ### Cam K
    assert list(data['cam_K'].shape) == [113, 3, 3]
    assert data['cam_K'].mean().item() == approx(72.28406524658203)
    assert data['cam_K'].var().item() == approx(7699.5048828125)
    
    assert data['cam_K'][60:69,0:2,1:2].mean().item() == approx(106.70677947998047)
    assert data['cam_K'][77:89,1:2,1:2].var().item() == approx(0.026782380416989326)
    
    # Scene 44796355
    data = dataset.get_camera_views('44796355')

    ### RGB
    assert list(data['rgb'].shape) == [144, 3, 256, 192]
    assert data['rgb'].mean().item() == approx(0.49844422936439514)
    assert data['rgb'].var().item() == approx(0.052545733749866486)
    
    assert data['rgb'][111:115,0:2,228:242,125:160].mean().item() == approx(0.4753442704677582)
    assert data['rgb'][94:143,0:1,108:239,46:131].var().item() == approx(0.050324659794569016)

    ### Depth
    assert list(data['depth_zbuffer'].shape) == [144, 256, 192]
    assert data['depth_zbuffer'].mean().item() == approx(2.2221503257751465)
    assert data['depth_zbuffer'].var().item() == approx(0.753137469291687)
    
    assert data['depth_zbuffer'][69:125,66:203,151:174].mean().item() == approx(2.3887500762939453)
    assert data['depth_zbuffer'][17:20,30:148,183:189].var().item() == approx(0.11931359767913818)

    ### Cam to world
    assert list(data['cam_to_world'].shape) == [144, 4, 4]
    assert data['cam_to_world'].mean().item() == approx(0.299155056476593)
    assert data['cam_to_world'].var().item() == approx(0.9597293734550476)

    assert data['cam_to_world'][97:120,2:3,2:3].mean().item() == approx(-0.03773859888315201)
    assert data['cam_to_world'][121:135,0:2,1:2].var().item() == approx(0.1889605075120926)

    ### Cam K
    assert list(data['cam_K'].shape) == [144, 3, 3]
    assert data['cam_K'].mean().item() == approx(71.85317993164062)
    assert data['cam_K'].var().item() == approx(7663.0654296875)
    
    assert data['cam_K'][55:126,0:2,0:1].mean().item() == approx(106.31748962402344)
    assert data['cam_K'][27:36,1:2,0:2].var().item() == approx(11980.294921875)

def test_snpp_camera_views():
    approx = functools.partial(ptapprox, rel=2e-3, abs=2e-3)

    ds = ScanNetPPDataset(SCANNETPP_DIRECTORY)
    frames = list(range(30*100))[::30][11:22]
    data = ds.get_camera_views("036bce3393", frames)

    # RGB
    assert list(data['rgb'].shape) == [11, 3, 480, 640]
    assert data['rgb'].mean().item() == approx(0.5180988907814026)
    assert data['rgb'].var().item() == approx(0.06683463603258133)
    
    assert data['rgb'][1:2,1:2,295:311,536:557].mean().item() == approx(0.4262488782405853)
    assert data['rgb'][1:8,0:1,160:204,628:629].var().item() == approx(0.033044662326574326)

    # Depth
    assert list(data['depth_zbuffer'].shape) == [11, 480, 640]
    assert data['depth_zbuffer'].mean().item() == approx(2.1898138523101807)
    assert data['depth_zbuffer'].var().item() == approx(0.7589995861053467)
    
    assert data['depth_zbuffer'][5:7,340:472,292:635].mean().item() == approx(1.4523074626922607)
    assert data['depth_zbuffer'][5:9,407:478,92:160].var().item() == approx(0.5101518630981445)

    # Cam to world
    assert list(data['cam_to_world'].shape) == [11, 4, 4]
    assert data['cam_to_world'].mean().item() == approx(0.8116238117218018)
    assert data['cam_to_world'].var().item() == approx(3.2991349697113037)
    
    assert data['cam_to_world'][5:7,1:2,1:2].mean().item() == approx(-0.28732001781463623)

    # Cam K
    assert list(data['cam_K'].shape) == [11, 3, 3]
    assert data['cam_K'].mean().item() == approx(167.99270629882812)
    assert data['cam_K'].var().item() == approx(40187.38671875)
    
    assert data['cam_K'][0:6,0:1,1:2].mean().item() == approx(0.0)
    assert data['cam_K'][0:3,0:1,1:2].var().item() == approx(0.0)
