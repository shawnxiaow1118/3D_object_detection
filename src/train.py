import
import argparse


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=100000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--output',dest='outputs')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def train():
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    log = Logger(outdir+'/log.txt',mode='a')

    ratios=np.array([0.5,1,2], dtype=np.float32)
    ## cfg.ratio, cfg.scales, cfg.bases
    scales=np.array([1,2,3],   dtype=np.float32)
    bases = make_bases(
        base_size = 16,
        ratios=ratios,
        scales=scales
    )
    num_bases = len(bases)
    stride = 8

    rgbs, tops, fronts, gt_labels, gt_boxes3d, top_imgs, front_imgs, lidars = load_dummy_datas()
    num_frames = len(rgbs)

    top_shape   = tops[0].shape
    front_shape = fronts[0].shape
    rgb_shape   = rgbs[0].shape
    top_feature_shape = (top_shape[0]//stride, top_shape[1]//stride)
    out_shape=(8,3)


    #-----------------------
    #check data
    if 0:
        fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
        draw_lidar(lidars[0], fig=fig)
        draw_gt_boxes3d(gt_boxes3d[0], fig=fig)
        mlab.show(1)
        cv2.waitKey(1)

    anchors, inside_inds =  make_anchors(bases, stride, top_shape[0:2], top_feature_shape[0:2])
    # inside_inds = np.arange(0,len(anchors),dtype=np.int32) 



    top_anchors     = tf.placeholder(shape=[None, 4], dtype=tf.int32,   name ='anchors'    )
    top_inside_inds = tf.placeholder(shape=[None   ], dtype=tf.int32,   name ='inside_inds')

    top_images   = tf.placeholder(shape=[None, *top_shape  ], dtype=tf.float32, name='top'  )
    front_images = tf.placeholder(shape=[None, *front_shape], dtype=tf.float32, name='front')
    rgb_images   = tf.placeholder(shape=[None, *rgb_shape  ], dtype=tf.float32, name='rgb'  )
    top_rois     = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='top_rois'   ) #<todo> change to int32???
    front_rois   = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='front_rois' )
    rgb_rois     = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='rgb_rois'   )


    # rpn_scores, rpn_probs, rpn_ deltas = rpn()

    top_features, top_scores, top_probs, top_deltas, proposals, proposal_scores = \
        top_feature_net(top_images, top_anchors, top_inside_inds, num_bases)

    front_features = front_feature_net(front_images)
    rgb_features   = rgb_feature_net(rgb_images)

    fuse_scores, fuse_probs, fuse_deltas = \
        fusion_net(
			( [top_features,     top_rois,     6,6,1./stride],
			  [front_features,   front_rois,   0,0,1./stride],  #disable by 0,0
			  [rgb_features,     rgb_rois,     6,6,1./stride],),
            num_class, out_shape) #<todo>  add non max suppression



    #loss ########################################################################################################
    top_inds     = tf.placeholder(shape=[None   ], dtype=tf.int32,   name='top_ind'    )
    top_pos_inds = tf.placeholder(shape=[None   ], dtype=tf.int32,   name='top_pos_ind')
    top_labels   = tf.placeholder(shape=[None   ], dtype=tf.int32,   name='top_label'  )
    top_targets  = tf.placeholder(shape=[None, 4], dtype=tf.float32, name='top_target' )
    top_cls_loss, top_reg_loss = rpn_loss(top_scores, top_deltas, top_inds, top_pos_inds, top_labels, top_targets)

    fuse_labels  = tf.placeholder(shape=[None            ], dtype=tf.int32,   name='fuse_label' )
    fuse_targets = tf.placeholder(shape=[None, *out_shape], dtype=tf.float32, name='fuse_target')
    fuse_cls_loss, fuse_reg_loss = rcnn_loss(fuse_scores, fuse_deltas, fuse_labels, fuse_targets)
    rgb_reg_loss = rgb_loss(rgb_scores, rgb_deltas, rgb_labels, rgb_targets)


    #solver
    l2 = l2_regulariser(decay=0.0005)
    learning_rate = tf.placeholder(tf.float32, shape=[])
    solver = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    #solver_step = solver.minimize(top_cls_loss+top_reg_loss+l2)
    solver_step = solver.minimize(top_cls_loss+top_reg_loss+fuse_cls_loss+0.1*fuse_reg_loss+l2)

    max_iter = 10000
    iter_debug=8

    # start training here  #########################################################################################
    log.write('epoch     iter    rate   |  top_cls_loss   reg_loss   |  fuse_cls_loss  reg_loss  |  \n')
    log.write('-------------------------------------------------------------------------------------\n')

    num_ratios=len(ratios)
    num_scales=len(scales)
    fig, axs = plt.subplots(num_ratios,num_scales)

    sess = tf.InteractiveSession()
    with sess.as_default():
        sess.run( tf.global_variables_initializer(), { IS_TRAIN_PHASE : True } )
        summary_writer = tf.summary.FileWriter(out_dir+'/tf', sess.graph)
        saver  = tf.train.Saver()

        batch_top_cls_loss =0
        batch_top_reg_loss =0
        batch_fuse_cls_loss=0
        batch_fuse_reg_loss=0
        for iter in range(max_iter):
            epoch=1.0*iter
            rate=0.05


            ## generate train image -------------
            # idx = np.random.choice(num_frames)     #*10   #num_frames)  #0
            batch_top_images    = tops[idx].reshape(1,*top_shape)
            batch_front_images  = fronts[idx].reshape(1,*front_shape)
            batch_rgb_images    = rgbs[idx].reshape(1,*rgb_shape)

            batch_gt_labels    = gt_labels[idx]
            batch_gt_boxes3d   = gt_boxes3d[idx]
            batch_gt_top_boxes = box3d_to_top_box(batch_gt_boxes3d)


			## run propsal generation ------------
            fd1={
                top_images:      batch_top_images,
                top_anchors:     anchors,
                top_inside_inds: inside_inds,

                learning_rate:   rate,
                IS_TRAIN_PHASE:  True
            }
            batch_proposals, batch_proposal_scores, batch_top_features = sess.run([proposals, proposal_scores, top_features],fd1)

            ## generate  train rois  ------------
            batch_top_inds, batch_top_pos_inds, batch_top_labels, batch_top_targets  = \
                rpn_target ( anchors, inside_inds, batch_gt_labels,  batch_gt_top_boxes)

            batch_top_rois, batch_fuse_labels, batch_fuse_targets  = \
                 rcnn_target(  batch_proposals, batch_gt_labels, batch_gt_top_boxes, batch_gt_boxes3d )

            batch_rois3d	 = project_to_roi3d    (batch_top_rois)
            # batch_front_rois = project_to_front_roi(batch_rois3d  )
            batch_rgb_rois   = project_to_rgb_roi  (batch_rois3d  )


            ##debug gt generation
            # if 1 and iter%iter_debug==0:
            #     top_image = top_imgs[idx]
            #     rgb       = rgbs[idx]

            #     img_gt     = draw_rpn_gt(top_image, batch_gt_top_boxes, batch_gt_labels)
            #     img_label  = draw_rpn_labels (top_image, anchors, batch_top_inds, batch_top_labels )
            #     img_target = draw_rpn_targets(top_image, anchors, batch_top_pos_inds, batch_top_targets)
            #     #imshow('img_rpn_gt',img_gt)
            #     #imshow('img_rpn_label',img_label)
            #     #imshow('img_rpn_target',img_target)

            #     img_label  = draw_rcnn_labels (top_image, batch_top_rois, batch_fuse_labels )
            #     img_target = draw_rcnn_targets(top_image, batch_top_rois, batch_fuse_labels, batch_fuse_targets)
            #     #imshow('img_rcnn_label',img_label)
            #     imshow('img_rcnn_target',img_target)


            #     img_rgb_rois = draw_boxes(rgb, batch_rgb_rois[:,1:5], color=(255,0,255), thickness=1)
            #     imshow('img_rgb_rois',img_rgb_rois)

            #     cv2.waitKey(1)

            ## run classification and regression loss -----------
            fd2={
				**fd1,

                top_images: batch_top_images,
                front_images: batch_front_images,
                rgb_images: batch_rgb_images,

				top_rois:   batch_top_rois,
                front_rois: batch_front_rois,
                rgb_rois:   batch_rgb_rois,

                top_inds:     batch_top_inds,
                top_pos_inds: batch_top_pos_inds,
                top_labels:   batch_top_labels,
                top_targets:  batch_top_targets,

                fuse_labels:  batch_fuse_labels,
                fuse_targets: batch_fuse_targets,
            }
            #_, batch_top_cls_loss, batch_top_reg_loss = sess.run([solver_step, top_cls_loss, top_reg_loss],fd2)


            _, batch_top_cls_loss, batch_top_reg_loss, batch_fuse_cls_loss, batch_fuse_reg_loss = \
               sess.run([solver_step, top_cls_loss, top_reg_loss, fuse_cls_loss, fuse_reg_loss],fd2)

            log.write('%3.1f   %d   %0.4f   |   %0.5f   %0.5f   |   %0.5f   %0.5f  \n' %\
				(epoch, iter, rate, batch_top_cls_loss, batch_top_reg_loss, batch_fuse_cls_loss, batch_fuse_reg_loss))
            if iter%iter_debug==0:
                top_image = top_imgs[idx]
                rgb       = rgbs[idx]

                batch_top_probs, batch_top_scores, batch_top_deltas  = \
                    sess.run([ top_probs, top_scores, top_deltas ],fd2)

                batch_fuse_probs, batch_fuse_deltas = \
                    sess.run([ fuse_probs, fuse_deltas ],fd2)

                #batch_fuse_deltas=0*batch_fuse_deltas #disable 3d box prediction
                probs, boxes3d = rcnn_nms(batch_fuse_probs, batch_fuse_deltas, batch_rois3d, threshold=0.5)


                ## show rpn score maps
                p = batch_top_probs.reshape( *(top_feature_shape[0:2]), 2*num_bases)
                for n in range(num_bases):
                    r=n%num_scales
                    s=n//num_scales
                    pn = p[:,:,2*n+1]*255
                    axs[s,r].cla()
                    axs[s,r].imshow(pn, cmap='gray', vmin=0, vmax=255)
                plt.pause(0.01)

				## show rpn(top) nms
                img_rpn     = draw_rpn    (top_image, batch_top_probs, batch_top_deltas, anchors, inside_inds)
                img_rpn_nms = draw_rpn_nms(top_image, batch_proposals, batch_proposal_scores)
                #imshow('img_rpn',img_rpn)
                imshow('img_rpn_nms',img_rpn_nms)
                cv2.waitKey(1)

                ## show rcnn(fuse) nms
                img_rcnn     = draw_rcnn (top_image, batch_fuse_probs, batch_fuse_deltas, batch_top_rois, batch_rois3d,darker=1)
                img_rcnn_nms = draw_rcnn_nms(rgb, boxes3d, probs)
                imshow('img_rcnn',img_rcnn)
                imshow('img_rcnn_nms',img_rcnn_nms)
                cv2.waitKey(1)

            # save: ------------------------------------
            if iter%500==0:
                #saver.save(sess, out_dir + '/check_points/%06d.ckpt'%iter)  #iter
                saver.save(sess, out_dir + '/check_points/snap.ckpt')  #iter
