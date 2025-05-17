# _*_ conding : utf-8 _*_
# @Time : 2025/4/9 14:27
# @Author : yang_4_ever
# @File : utils
# @Project : FinalWMsys
# 语言字典
LANGUAGES = {
    "en": {
        "window_title": "DNN Model Watermarking System",
        "operations": "Operations",
        "output": "Output",
        "model_ops": "Model Operations",
        "model_ops_help": "1. Load Pretrained Model: Load a .pth file with ResNet50 architecture\n"
                          "2. Save Model: Save the current model with watermark\n"
                          "3. Load Model: Load a previously saved watermarked model",
        "load_pretrained": "Load Pretrained Model",
        "save_model": "Save Model",
        "load_model": "Load Model",
        "watermark_gen": "Watermark Generation",
        "watermark_gen_help": "Generate a unique watermark key based on owner information.\n"
                             "This will also create a trigger set of random images with labels.",
        "owner_info": "Owner Information:",
        "owner_placeholder": "Enter owner information here...",
        "gen_watermark": "Generate Watermark",
        "whitebox": "White-box Watermark",
        "whitebox_help": "Embed: Modifies layer4 parameters to encode the watermark key\n"
                         "Extract: Reads layer4 parameters to recover the watermark\n"
                         "Strength: Controls how much the parameters are modified (0.01-1.0)",
        "embed_strength": "Embedding Strength:",
        "embed_watermark": "Embed Watermark",
        "extract_watermark": "Extract Watermark",
        "watermark_generated": "Watermark generated successfully!",
        "watermark_key": "Watermark Key (first 16 chars): {}...",
        "no_owner_info": "Please enter owner information",
        "original_key": "Original Key (first 16 chars): {}",
        "extracted_key": "Extracted Key: {}",
        "watermark_match": "Watermark Match: {}",
        "no_key": "No watermark key found in loaded model",
        "watermark_verify": "Watermark Verification",
        "enter_owner_verify": "Enter owner information for verification:",
        "verify_watermark": "Verify Watermark",
        "verify_success": "✅ Watermark VERIFIED (Owner matched)",
        "verify_fail": "❌ Watermark INVALID (Owner mismatch)",
        "no_watermark": "⚠️ No watermark detected in model",
        "regen_watermark": "⚠️ Watermark regenerated from owner info",
        "run_tests": "Run All Tests",
        "performance_report": "Performance Report",
        "load_test_data": "Load Test Dataset",
        "performance_test": "Performance Testing",
        "performance_test_help": "Run comprehensive tests to evaluate:\n"
                                "1. Model accuracy before/after watermarking\n"
                                "2. Watermark extraction success rate\n"
                                "3. Robustness against common attacks",
        "running_tests": "Running performance tests...",
        "original_accuracy": "Original Accuracy",
        "extraction_success": "Watermark Extraction Success Rate",
        "robustness_report": "Robustness Test Results",
        "accuracy": "Model Accuracy",
        "extraction_rate": "Watermark Extraction Rate",
        "no_test_data": "Please load test data first",
        "no_model_loaded": "Please load a model first",
        "test_data_loaded": "Test dataset loaded successfully",
        "test_data_error": "Error loading test dataset",
        "test_results": "Test Results",
        "help_button": "Help",
        "close_button": "Close",
        "cancel_button": "Cancel",
        "confirm_button": "Confirm",
        "success_title": "Success",
        "error_title": "Error",
        "warning_title": "Warning"
    },
    "zh": {
        "window_title": "DNN模型水印系统",
        "operations": "操作面板",
        "output": "输出信息",
        "model_ops": "模型操作",
        "model_ops_help": "1. 加载预训练模型: 加载ResNet50架构的.pth文件\n"
                          "2. 保存模型: 保存当前带水印的模型\n"
                          "3. 加载模型: 加载之前保存的水印模型",
        "load_pretrained": "加载预训练模型",
        "save_model": "保存模型",
        "load_model": "加载模型",
        "watermark_gen": "水印生成",
        "watermark_gen_help": "基于所有者信息生成唯一水印密钥\n"
                             "同时会创建一组带有标签的随机图像作为触发集",
        "owner_info": "所有者信息:",
        "owner_placeholder": "在此输入所有者信息...",
        "gen_watermark": "生成水印",
        "whitebox": "白盒水印",
        "whitebox_help": "嵌入: 修改layer4参数以编码水印密钥\n"
                         "提取: 读取layer4参数以恢复水印\n"
                         "强度: 控制参数修改程度(0.01-1.0)",
        "embed_strength": "嵌入强度:",
        "embed_watermark": "嵌入水印",
        "extract_watermark": "提取水印",
        "watermark_generated": "水印生成成功!",
        "watermark_key": "水印密钥(前16位): {}...",
        "no_owner_info": "请输入所有者信息",
        "original_key": "原始密钥(前16位): {}",
        "extracted_key": "提取的密钥: {}",
        "watermark_match": "水印匹配: {}",
        "no_key": "加载的模型中未找到水印密钥",
        "watermark_verify": "水印验证",
        "enter_owner_verify": "输入所有者信息进行验证:",
        "verify_watermark": "验证水印",
        "verify_success": "✅ 水印验证通过 (所有者匹配)",
        "verify_fail": "❌ 水印验证失败 (所有者不匹配)",
        "no_watermark": "⚠️ 模型中未检测到水印",
        "regen_watermark": "⚠️ 已从所有者信息重新生成水印",
        "run_tests": "执行所有测试",
        "performance_report": "性能报告",
        "load_test_data": "加载测试数据集",
        "performance_test": "性能测试",
        "performance_test_help": "运行综合测试评估:\n"
                                "1. 水印前后的模型准确率\n"
                                "2. 水印提取成功率\n"
                                "3. 对常见攻击的鲁棒性",
        "running_tests": "正在运行性能测试...",
        "original_accuracy": "原始准确率",
        "extraction_success": "水印提取成功率",
        "robustness_report": "鲁棒性测试结果",
        "accuracy": "模型准确率",
        "extraction_rate": "水印提取率",
        "no_test_data": "请先加载测试数据",
        "no_model_loaded": "请先加载模型",
        "test_data_loaded": "测试数据集加载成功",
        "test_data_error": "加载测试数据集出错",
        "test_results": "测试结果",
        "help_button": "帮助",
        "close_button": "关闭",
        "cancel_button": "取消",
        "confirm_button": "确认",
        "success_title": "成功",
        "error_title": "错误",
        "warning_title": "警告"
    }
}