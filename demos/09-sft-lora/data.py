"""人工构造的中文客服分类数据，三个 split 在训练前固定。"""
import json


SYSTEM = "你是一个客服消息分类助手。"


def row(text, category):
    return {"user": text, "assistant": json.dumps({"类别": category}, ensure_ascii=False, separators=(",", ":"))}


TRAIN = [
    row("商品坏了，我要退款。", "退款"),
    row("我不想要了，怎么退钱？", "退款"),
    row("买错了，请帮我退掉这笔订单。", "退款"),
    row("收到的杯子破了，申请退货退款。", "退款"),
    row("尺码不合适，我想退货。", "退款"),
    row("订单取消了，钱什么时候退给我？", "退款"),
    row("东西有质量问题，请退回货款。", "退款"),
    row("重复买了两份，其中一份要退款。", "退款"),
    row("我的快递到哪里了？", "物流"),
    row("订单什么时候发货？", "物流"),
    row("请帮我查一下运单号。", "物流"),
    row("包裹三天没更新了，能查一下吗？", "物流"),
    row("今天可以送到吗？", "物流"),
    row("快递显示签收，但我还没拿到。", "物流"),
    row("我想确认一下配送进度。", "物流"),
    row("商品发出以后大概几天能到？", "物流"),
]
VALIDATION = [
    row("这件衣服我不要了，帮我退回付款。", "退款"),
    row("收到的东西不能用，我需要退钱。", "退款"),
    row("帮我看看包裹现在在哪个城市。", "物流"),
    row("下单两天了，什么时候寄出？", "物流"),
]
TEST = [
    row("鞋子太小，能办理退货并退钱吗？", "退款"),
    row("取消购买后，我想知道退款进度。", "退款"),
    row("运单一直停在中转站，什么时候继续配送？", "物流"),
    row("请告诉我这单预计的送达时间。", "物流"),
]


def messages(example, with_answer=False):
    result = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": example["user"]}]
    if with_answer:
        result.append({"role": "assistant", "content": example["assistant"]})
    return result


def tokenize_example(tokenizer, example, max_length=256):
    prefix = tokenizer.apply_chat_template(messages(example), tokenize=True, add_generation_prompt=True)
    ids = tokenizer.apply_chat_template(messages(example, True), tokenize=True, add_generation_prompt=False)
    # 直接对完整 chat template tokenize，避免分别编码文本后拼接导致边界变化。
    if ids[:len(prefix)] != prefix:
        raise ValueError("当前 tokenizer 的训练文本与生成前缀不对齐，需要重新确定答案边界。")
    if len(ids) > max_length:
        raise ValueError(f"样本有 {len(ids)} tokens，超过 {max_length}；本课不静默截断答案。")
    if len(ids) <= len(prefix):
        raise ValueError("样本没有可监督的 assistant tokens。")
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prefix) + ids[len(prefix):]}
