"""
  @Author:fengzhipeng
  @Time:2026/7/31
  @Desc:
"""
from pathlib import Path

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState


class EntryNode(BaseNode):
    """入口节点
    根据上传文件后缀,来决定走哪个分支

    """

    name = "entry_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1.获取文件以及路径
        import_file_path_obj = state.get("import_file_path", "")
        file_dir = state.get("file_dir", "")
        # 2.判断上传文件是否为空
        if not import_file_path_obj:
            raise ValidationError(f"import_file_path的键值为空", self.name)
        # 3. 获取文件后缀,并变成小写
        path = Path(import_file_path_obj)
        if not path.exists():
            raise ValidationError("文件不存在", self.name)
        suffix = path.suffix.lower()
        # 4.不同文件类型做不同状态修改
        if suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
        if suffix == (".md"):
            state["is_md_read_enabled"] = True
        # 5.获取文件标题,不带扩展名
        state["file_title"] = path.stem
        # 6.更新咋=状态并返回状态
        return state