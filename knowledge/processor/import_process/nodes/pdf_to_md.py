import json
import subprocess
import time
from typing import Tuple
from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError, PdfConversionError
from knowledge.processor.import_process.state import ImportGraphState


class PdfToMdNode(BaseNode):
    """
    pdf转md
    """

    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        用MinerU 将上传的pdf转为md
        :param stae:
                state = {
                    "is_pdf_read_enabled": True,
                    "is_md_read_enabled": False,
                    "import_file_path": D:\\BaiduNetdiskDownload\\11_尚硅谷AI全能开发技术之项目【掌柜智库】\\2.资料\\pdf文档\\doc\\万用表RS-12的使用.pdf,
                    "file_dir": D:\\Python-Project\\temp_dir
                }
        :return:

        """
        #1.参数校验
        import_file_path_obj, file_dir_path_obj = self._validate_state_inputs_path(state)
        #2.用mineru命令将pdf转换成md
        processed_code = self._excute_mineru(import_file_path_obj, file_dir_path_obj)
        if processed_code != 0:
            raise PdfConversionError("MinerU解析PDF失败", self.name)
        #3.获取md文件路径
        md_path = self._get_md_paths(import_file_path_obj, file_dir_path_obj)
        #4.更新状态并返回状态
        state['md_path'] = md_path
        return state

    def _validate_state_inputs_path(self, state: ImportGraphState) -> Tuple[Path, Path]:
        """校验输入参数:文件路径,输出目录"""
        import_file_path = state.get("import_file_path", "")
        file_dir = state.get("file_dir", "")
        if not import_file_path:
            raise ValidationError("输入文件路径为空", self.name)
        import_file_path_obj = Path(import_file_path) #转换成Path对象

        if not import_file_path_obj.exists():
            raise ValidationError("输入文件路径不存在", self.name)
        if not file_dir:
            file_dir = import_file_path_obj.parent      #如果输出路径不存在，则使用输入父路径
        file_dir_path_obj = Path(file_dir) #转换成Path对象

        return import_file_path_obj, file_dir_path_obj

    def _excute_mineru(self, import_file_path_obj, file_dir_path_obj) -> int:

        """
                执行 MinerU 命令

                Args:
                    import_file_path: 解析的文件路径
                    file_dir_path: 解析后的文件目录

                Returns:
                    命令返回码（0 表示成功）
                """
        self.log_step("step2", "执行MinerU解析PDF")

        # 1. 构建命令行
        cmd = [
            "mineru",
            "-p",
            str(import_file_path_obj),
            "-o",
            str(file_dir_path_obj),
            "--source",
            "local",
            "--backend",
            "pipeline"
        ]

        process_start_time = time.time()

        # 2. 执行命令行(子进程执行命令行)
        proc = subprocess.Popen(
            args=cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            errors="replace",  # 遇到乱码时替换
            text=True,  # 输出的内容是字符串 不是字节
            encoding="utf-8",  # 用指定的中文字符集进行编解码
            bufsize=1  # 按行缓冲，只要缓冲区一行满了就输出
        )

        # 3. 获取日志信息
        for line in proc.stdout:
            self.logger.info(f"执行MinerU产生的日志：{line}")

        # 4. 等待子进程做完
        processed_code = proc.wait()

        process_end_time = time.time()
        if processed_code == 0:
            self.logger.info(
                f"MinerU成功解析PDF文件：{import_file_path_obj.name} "
                f"耗时:{process_end_time - process_start_time:.2f}s"
            )
        else:
            self.logger.error(f"MinerU解析PDF文件：{import_file_path_obj.name}失败")

        # 5. 返回状态码
        return processed_code

    def _get_md_paths(self, import_file_path_obj: Path, file_dir_path_obj: Path) -> str:
        """
        获取转换结果路径

        MinerU 输出目录结构:
        file_dir_path/
          └── 文件名/
               └── hybrid_auto/
                    ├── 文件名.md
                    └── images/

        Args:
            import_file_path: PDF 文件 Path 对象
            file_dir_path: 输出目录 Path 对象

        Returns:
            Markdown 文件路径字符串
        """
        file_name = import_file_path_obj.stem
        md_path = file_dir_path_obj / file_name / "auto" / f"{file_name}.md"
        return str(md_path)



if __name__ == '__main__':
    setup_logging()
    pdf_to_md_node = PdfToMdNode()

    pdf_to_md_node_init_state = {
        "import_file_path": "D:\\BaiduNetdiskDownload\\11_尚硅谷AI全能开发技术之项目【掌柜智库】\\2.资料\\pdf文档\\doc\\万用表RS-12的使用.pdf",
        "file_dir": "D:\\Python-Project\\temp_dir"
    }
    processed_result = pdf_to_md_node.process(pdf_to_md_node_init_state)
	#indent=4 缩进
    #ensure_ascii=False确保不出现乱码
    print(json.dumps(processed_result, indent=4, ensure_ascii=False))