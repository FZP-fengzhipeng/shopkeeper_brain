import json
from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState


class EntryNode(BaseNode):
    """入口节点
    根据上传文件后缀,来决定走哪个分支

    """

    name = "entry_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        #1.获取文件以及路径
        self.log_step("步骤1: 获取文件以及路径")
        import_file_path = state.get("import_file_path", "")
        file_dir = state.get("file_dir", "")

        #2.判断上传文件是否为空
        self.log_step("步骤2: 判断上传文件是否为空")
        if not import_file_path:
            raise ValidationError(f"import_file_path的键值为空", self.name)

        #3. 获取文件后缀,并变成小写
        self.log_step("步骤3: 获取文件后缀,并变成小写")
        path = Path(import_file_path)
        if not path.exists():
            raise ValidationError(f"{import_file_path}文件不存在", self.name)
        suffix = path.suffix.lower()

        #4.不同文件类型做不同状态修改
        self.log_step("步骤4: 不同文件类型做不同状态修改")
        if suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state['pdf_path'] = import_file_path
            #视频写的["is_pdf_enable"]
        elif suffix == (".md"):
            state["is_md_read_enabled"] = True
            state['md_path'] = import_file_path
        else:
            raise ValidationError(f"不支持的文件类型: {suffix}", self.name)

        #5.获取文件标题,不带扩展名
        self.log_step("步骤5: 获取文件标题,不带扩展名")
        state["file_title"] = path.stem

        #6.更新咋=状态并返回状态
        self.log_step("步骤6: 更新状态并返回状态")
        return state
if __name__ == "__main__":
    setup_logging()
    import_file_path = "D:\\BaiduNetdiskDownload\\11_尚硅谷AI全能开发技术之项目【掌柜智库】\\2.资料\\pdf文档\\doc\\万用表RS-12的使用.pdf"
    file_dir = "D:\\Python-Project\\temp_dir"
    state = {
        "is_pdf_read_enabled": False,
        "is_md_read_enabled": False,
        "import_file_path": import_file_path,
        "file_dir": file_dir
    }

    entry = EntryNode()

    processed_state = entry(state)  # __call__ -> process()

    #entry.process(state)

    print(json.dumps(processed_state, indent=4, ensure_ascii=False))