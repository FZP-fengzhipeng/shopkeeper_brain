import os


import uvicorn
from fastapi import FastAPI, UploadFile, File, Depends, BackgroundTasks
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

from knowledge.core.deps import get_import_file_service
from knowledge.core.paths import get_front_page_dir
from knowledge.processor.query_process.base import setup_logging
from knowledge.schema.upload_schema import UploadResponse, TaskStatusResponse
from knowledge.service.import_file_service import ImportFileService
from knowledge.utils.task_util import get_task_info


def register_router(app):

    @app.get("/import")
    async def import_page():
        return FileResponse(os.path.join(get_front_page_dir(),"import.html"))

    @app.post("/upload", response_model=UploadResponse)
    async def upload_file_enpoint(background_tasks:BackgroundTasks,# 别让用户等,后台运行
                                  service:ImportFileService=Depends(get_import_file_service),#依赖注入ImportFileService service
                                  file:UploadFile = File(...)):#拿到文件
        #get_import_file_service不加(),延迟加载,什么时候用才加载.加了(),表示立即调用方法
        """上传文件接口"""
        #1.文件上传,双写.写本地并且上传minio
        task_id,file_dir,import_file_path = service.process_upload_file(file)

        #2.开启langgraph流程
        #service.process_langgraph(task_id,file_dir,import_file_path)
        background_tasks.add_task(service.run_langgraph_import,task_id,file_dir,import_file_path)
        return UploadResponse(task_id=task_id,message="文件上传成功,请等待导入流程完成")

    @app.get("/status/{task_id}",response_model=TaskStatusResponse)
    async def get_statu(task_id:str):
        """
        根据任务ID 获取任务状态
        :param task_id:
        :return:
        """
        task_info = get_task_info(task_id)
        return  TaskStatusResponse(**task_info)

def create_app():
    setup_logging()
    app = FastAPI(description="知识库导入",version="v1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],#对客户端访问来源限制
        allow_credentials=False, #对请求cookie或Authorization头进行限制,如果设置为True,其他三个不能使用*
        allow_methods=["*"],#对请求方法限制
        allow_headers=["*"],# 对请求头限制
    )

    # 挂载前端静态资源，  image,css
    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))
    register_router(app)
    return  app


if __name__ == "__main__":
    uvicorn.run(app=create_app(), host="0.0.0.0", port=8000,log_level="info")