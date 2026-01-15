import graphviz

dot = graphviz.Digraph('AestheticModel', comment='Model Structure')
dot.attr(rankdir='LR', nodesep='0.6', ranksep='0.8')
dot.attr('node', fontname='Segoe UI', fontsize='14', shape='box', style='filled')
dot.attr('edge', fontname='Consolas', fontsize='10')

# 1. Inputs Cluster
with dot.subgraph(name='cluster_inputs') as c:
    c.attr(label='Inputs', style='dashed', color='gray')
    c.node('Image', 'Image\n[Batch, C, H, W]', fillcolor='#e3f2fd', shape='note')
    c.node('Prompt', 'Prompt\n(Token IDs)\n[Batch, Seq]', fillcolor='#fff3e0', shape='note')

# 2. Encoders Cluster
with dot.subgraph(name='cluster_encoders') as c:
    c.attr(label='Encoders (Frozen/Finemode)', style='dashed', color='blue')
    c.node('VisionEncoder', 'Vision Encoder\nViT-L-14', fillcolor='#bbdefb')
    c.node('TextEncoder', 'Text Encoder\nXLM-RoBERTa-Large', fillcolor='#ffe0b2')

# 3. Vision Projector Cluster
with dot.subgraph(name='cluster_vision_proj') as c:
    c.attr(label='Vision Projector', style='dashed', color='orange')
    c.node('Vis_Linear', 'Linear\n1024 -> 768\n(通过训练一个矩阵然后和1024相乘实现)', fillcolor='#ffcc80') 
    c.node('Vis_LN', 'LayerNorm\n(作用: 归一化特征，稳定训练)', fillcolor='#ffcc80')
    c.node('Vis_GELU', 'GELU\n(实际上是非线性的改变对齐两个模态的结构)', fillcolor='#ffcc80')
    # Internal edges
    c.edge('Vis_Linear', 'Vis_LN')
    c.edge('Vis_LN', 'Vis_GELU')

# 4. Text Projector Cluster
with dot.subgraph(name='cluster_text_proj') as c:
    c.attr(label='Text Projector', style='dashed', color='orange')
    c.node('Txt_Linear', 'Linear\n1024 -> 768\n(通过训练一个矩阵然后和1024相乘实现)', fillcolor='#ffcc80')
    c.node('Txt_LN', 'LayerNorm\n(作用: 归一化特征，稳定训练)', fillcolor='#ffcc80')
    c.node('Txt_GELU', 'GELU\n(实际上是非线性的改变对齐两个模态的结构)', fillcolor='#ffcc80')
    # Internal edges
    c.edge('Txt_Linear', 'Txt_LN')
    c.edge('Txt_LN', 'Txt_GELU')

# 5. Fusion
dot.node('Concat', 'Concatenation\n(Vision + Text)', fillcolor='#e1bee7', shape='box')

# 6. Multi-Head Prediction Cluster
with dot.subgraph(name='cluster_heads') as c:
    c.attr(label='Multi-Head Prediction (MLP)', style='dashed', color='green')
    
    heads = [
        ('total_score', 'Total Score'),
        ('composition', 'Composition'),
        ('color', 'Color'),
        ('lighting', 'Lighting'),
        ('subject_clarity', 'Subject Clarity'),
        ('creativity', 'Creativity'),
        ('commercial_appeal', 'Commercial Appeal')
    ]
    
    for head_id, head_label in heads:
        head_node_name = f'Head_{head_id}'
        out_node_name = f'Out_{head_id}'
        c.node(head_node_name, f'{head_label} Head\n[Linear->ReLU->Dropout->Linear]', fillcolor='#c8e6c9')
        c.node(out_node_name, f'{head_label}\nScore', fillcolor='#a5d6a7', shape='ellipse')
        c.edge(head_node_name, out_node_name, label='[B, 1]')

# Main Edges
dot.edge('Image', 'VisionEncoder')
dot.edge('Prompt', 'TextEncoder')

dot.edge('VisionEncoder', 'Vis_Linear', label='[B, 1024]')
dot.edge('TextEncoder', 'Txt_Linear', label='[B, 1024]')

dot.edge('Vis_GELU', 'Concat', label='[B, 768]')
dot.edge('Txt_GELU', 'Concat', label='[B, 768]')

for head_id, _ in heads:
    dot.edge('Concat', f'Head_{head_id}', label='[B, 1536]')

# Save source
dot.save('model_structure.dot')




