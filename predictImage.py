#!/usr/bin/python
#coding:utf-8

from django.http import HttpResponse
from predict import *
import json

def recognition(request):
    if request.method == 'POST':
        file_obj = request.FILES.get('image')
        import os
        imagePath = os.path.join('static', 'pic', file_obj.name)
        f = open(imagePath, 'wb')
        print(file_obj,type(file_obj))
        for chunk in file_obj.chunks():
            f.write(chunk)
        f.close()
        imageType = predictImage(imagePath)
        os.remove(imagePath)
    if imageType == 0:
        data = {
            'resultsCode': str(imageType),
        }
        resp = {
            'status': '0',
            'msg': '成功',
            'data': data,
        }
    return HttpResponse(json.dumps(resp, {'ensure_ascii':False}), content_type="application/json")
