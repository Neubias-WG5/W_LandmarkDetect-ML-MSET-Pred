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
"""
Given the classifier clf, this function will try to find the landmark on the
image current
"""


def searchpoint_cytomine(repository, current, clf, mx, my, cm, depths, window_size, feature_type, feature_parameters,
						 image_type,
						 npred):
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

	return np.median(maxx), (height + 1) - np.median(maxy)


def str2bool(v):
	return v.lower() in ("yes", "true", "t", "1")

def find_by_attribute(att_fil, attr, val):
	return next(iter([i for i in att_fil if hasattr(i, attr) and getattr(i, attr) == val]), None)

def main():
	with CytomineJob.from_cli(sys.argv) as conn:
		base_path = "{}".format(os.getenv("HOME"))
		working_path = os.path.join(base_path, str(conn.job.id))
		in_path = os.path.join(working_path, "in/")
		out_path = os.path.join(working_path, "out/")

		tr_working_path = os.path.join(base_path, str(conn.parameters.model_to_use))
		tr_out_path = os.path.join(tr_working_path, "out/")

		if not os.path.exists(working_path):
			os.makedirs(working_path)
			os.makedirs(in_path)

		images = ImageInstanceCollection().fetch_with_filter("project", conn.parameters.cytomine_id_project)
		list_imgs = []
		if conn.parameters.images_to_predict == 'all':
			for image in images:
				list_imgs.append(int(image.id))
				image.dump(os.path.join(in_path, '%d.jpg' % (image.id)))
		else:
			list_imgs = [int(id_img) for id_img in conn.parameters.images_to_predict.split(',')]
			for image in images:
				if image.id in list_imgs:
					image.dump(os.path.join(in_path, '%d.jpg' % (image.id)))

		annotation_collection = AnnotationCollection()
		train_job = Job().fetch(conn.parameters.model_to_use)
		properties = PropertyCollection(train_job).fetch()
		str_terms = ""
		for prop in properties:
			if prop.fetch(key='id_terms')!=None:
				str_terms = prop.fetch(key='id_terms').value
		term_list = [int(x) for x in str_terms.split(' ')]
		attached_files = AttachedFileCollection(train_job).fetch()

		for id_term in conn.monitor(term_list, start=10, end=90, period = 0.05, prefix="Finding landmarks for terms..."):
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
				(x, y) = searchpoint_cytomine(in_path, id_img, model, mx, my, cm, 1. / (2. ** np.arange(parameters_hash['model_depth'])), parameters_hash['window_size'], parameters_hash['feature_type'], feature_parameters, 'jpg', parameters_hash['model_npred'])
				circle = Point(x, y)
				annotation_collection.append(Annotation(location=circle.wkt, id_image=id_img, id_terms=[id_term], id_project=conn.parameters.cytomine_id_project))

		annotation_collection.save()

if __name__ == "__main__":
	main()