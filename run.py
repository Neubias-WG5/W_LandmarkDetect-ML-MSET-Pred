# -*- coding: utf-8 -*-

#
# * Copyright (c) 2009-2016. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */


__author__ = "Vandaele Rémy <remy.vandaele@ulg.ac.be>"
__contributors__ = ["Marée Raphaël <raphael.maree@ulg.ac.be>"]
__copyright__ = "Copyright 2010-2016 University of Liège, Belgium, http://www.cytomine.be/"

from sklearn.externals import joblib
from shapely.geometry import Point
import sys, os

from ldmtools import *
from cytomine import cytomine, models, CytomineJob
from cytomine.models import *
from cytomine.models import Annotation, Job, ImageInstanceCollection, AnnotationCollection, Property, AttachedFileCollection, AttachedFile
import numpy as np
from neubiaswg5 import CLASS_LNDDET
from neubiaswg5.helpers import NeubiasJob, prepare_data, upload_data, upload_metrics, get_discipline
from neubiaswg5.helpers.data_upload import imwrite
"""
Given the classifier clf, this function will try to find the landmark on the
image current
"""


def searchpoint_cytomine(repository, current, clf, mx, my, cm, depths, window_size, feature_type, feature_parameters, image_type, npred):
	simage = readimage(repository, current, image_type)
	(height, width) = simage.shape

	P = np.random.multivariate_normal([mx, my], cm, npred)
	x_v = np.round(P[:, 0] * width)
	y_v = np.round(P[:, 1] * height)

	height = height - 1
	width = width - 1

	n = len(x_v)
	pos = 0

	maxprob = -1
	maxx = []
	maxy = []

	# maximum number of points considered at once in order to not overload the
	# memory.
	step = 100000

	for index in range(len(x_v)):
		xv = x_v[index]
		yv = y_v[index]
		if (xv < 0):
			x_v[index] = 0
		if (yv < 0):
			y_v[index] = 0
		if (xv > width):
			x_v[index] = width
		if (yv > height):
			y_v[index] = height

	while (pos < n):
		xp = np.array(x_v[pos:min(n, pos + step)])
		yp = np.array(y_v[pos:min(n, pos + step)])

		DATASET = build_dataset_image(simage, window_size, xp, yp, feature_type, feature_parameters, depths)
		pred = clf.predict_proba(DATASET)
		pred = pred[:, 1]
		maxpred = np.max(pred)
		if (maxpred >= maxprob):
			positions = np.where(pred == maxpred)
			positions = positions[0]
			xsup = xp[positions]
			ysup = yp[positions]
			if (maxpred > maxprob):
				maxprob = maxpred
				maxx = xsup
				maxy = ysup
			else:
				maxx = np.concatenate((maxx, xsup))
				maxy = np.concatenate((maxy, ysup))
		pos = pos + step

	return np.median(maxx), np.median(maxy), height, width


def str2bool(v):
	return v.lower() in ("yes", "true", "t", "1")

def find_by_attribute(att_fil, attr, val):
	return next(iter([i for i in att_fil if hasattr(i, attr) and getattr(i, attr) == val]), None)

def main():
	with NeubiasJob.from_cli(sys.argv) as conn:
		problem_cls = get_discipline(conn, default=CLASS_LNDDET)
		is_2d = True
		conn.job.update(status=Job.RUNNING, progress=0, statusComment="Initialization of the prediction phase")
		in_images, gt_images, in_path, gt_path, out_path, tmp_path = prepare_data(problem_cls, conn, is_2d=is_2d, **conn.flags)
		list_imgs = [int(image.rstrip('.tif')) for image in os.listdir(in_path) if image.endswith('.tif')]

		train_job = Job().fetch(conn.parameters.model_to_use)
		properties = PropertyCollection(train_job).fetch()
		str_terms = ""
		for prop in properties:
			if prop.fetch(key='id_terms')!=None:
				str_terms = prop.fetch(key='id_terms').value
		term_list = [int(x) for x in str_terms.split(' ')]
		attached_files = AttachedFileCollection(train_job).fetch()

		hash_pos = {}
		hash_size = {}
		for id_term in conn.monitor(term_list, start=10, end=70, period = 0.05, prefix="Finding landmarks for terms..."):
			model_file = find_by_attribute(attached_files, "filename", "%d_model.joblib"%id_term)
			model_filepath = os.path.join(in_path, "%d_model.joblib"%id_term)
			model_file.download(model_filepath, override=True)
			cov_file = find_by_attribute(attached_files, 'filename', '%d_cov.joblib'%id_term)
			cov_filepath = os.path.join(in_path, "%d_cov.joblib"%id_term)
			cov_file.download(cov_filepath, override=True)
			parameters_file = find_by_attribute(attached_files, 'filename', '%d_parameters.joblib'%id_term)
			parameters_filepath = os.path.join(in_path, '%d_parameters.joblib'%id_term)
			parameters_file.download(parameters_filepath, override=True)

			model = joblib.load(model_filepath)
			[mx, my, cm] = joblib.load(cov_filepath)
			parameters_hash = joblib.load(parameters_filepath)
			feature_parameters = None
			if parameters_hash['feature_type'] in ['haar', 'gaussian']:
				fparameters_file = find_by_attribute(attached_files, 'filename', "%d_fparameters.joblib"%id_term)
				fparametersl_filepath = os.path.join(in_path, "%d_fparameters.joblib"%id_term)
				fparameters_file.download(fparametersl_filepath, override=True)
				feature_parameters = joblib.load(fparametersl_filepath)
			for id_img in list_imgs:
				(x, y, height, width) = searchpoint_cytomine(in_path, id_img, model, mx, my, cm, 1. / (2. ** np.arange(parameters_hash['model_depth'])), parameters_hash['window_size'], parameters_hash['feature_type'], feature_parameters, 'tif', parameters_hash['model_npred'])
				if (not id_img in hash_size):
					hash_size[id_img] = (height, width)
					hash_pos[id_img] = []
				hash_pos[id_img].append(((id_term, x, y)))
		conn.job.update(status=Job.RUNNING, progress=95, statusComment="Uploading the results...")
		for id_img in list_imgs:
			(h, w) = hash_size[id_img]
			lbl_img = np.zeros((h, w), 'uint8')
			for (id_term, x, y) in hash_pos[id_img]:
				intx = int(x)
				inty = int(y)
				if lbl_img[inty, intx] > 0:
					(ys, xs) = np.where(lbl_img==0)
					dis = np.sqrt((ys-y)**2 + (xs-x)**2)
					j = np.argmin(dis)
					intx = int(xs[j])
					inty = int(ys[j])
				lbl_img[inty, intx] = id_term
			imwrite(path=os.path.join(out_path, '%d.tif'%id_img), image=lbl_img.astype(np.uint8), is_2d=is_2d)
		upload_data(problem_cls, conn, in_images, out_path, **conn.flags, is_2d=is_2d, monitor_params={"start": 70, "end": 90, "period": 0.1})
		conn.job.update(progress=90, statusComment="Computing and uploading metrics (if necessary)...")
		upload_metrics(problem_cls, conn, in_images, gt_path, out_path, tmp_path, **conn.flags)
		conn.job.update(status=Job.TERMINATED, progress=100, statusComment="Finished.")

if __name__ == "__main__":
	main()