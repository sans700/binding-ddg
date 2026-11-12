import argparse
import base64
import csv
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',default='results/run_001'); a=ap.parse_args()
    run=Path(a.run); figures=run/'figures'; figures.mkdir(exist_ok=True)
    audit=json.loads((run/'data_audit.json').read_text()); metrics=json.loads((run/'metrics.json').read_text())
    config=json.loads((run/'run_config.json').read_text()); runs=json.loads((run/'training_runs.json').read_text())
    with (run/'learning_curves.csv').open() as f: curves=list(csv.DictReader(f))
    with (run/'predictions.csv').open() as f: data=list(csv.DictReader(f))
    test=[r for r in data if r['split']=='test']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
    fig,axes=plt.subplots(1,len(runs),figsize=(11,3.25),sharey=True,constrained_layout=True)
    for ax,seedrun in zip(np.atleast_1d(axes),runs):
        c=[r for r in curves if int(r['seed'])==seedrun['seed']]
        ax.plot([int(r['epoch']) for r in c],[float(r['train_rmse']) for r in c],label='Training',color='#2D658A')
        ax.plot([int(r['epoch']) for r in c],[float(r['validation_rmse']) for r in c],label='Validation',color='#C3753D')
        ax.axvline(seedrun['best_epoch'],ls=':',color='#777777',label='Selected epoch')
        ax.set(title=f"Seed {seedrun['seed']}",xlabel='Epoch'); ax.grid(alpha=.18)
    np.atleast_1d(axes)[0].set_ylabel('RMSE (kcal/mol)'); np.atleast_1d(axes)[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Early stopping is based on validation only',fontsize=12)
    fig.savefig(figures/'learning_curves.png',dpi=180); plt.close(fig)
    y=np.array([float(r['ddg_kcal_mol']) for r in test]); pred=np.array([float(r['mlp_prediction']) for r in test])
    fig,ax=plt.subplots(figsize=(5.8,5.2),constrained_layout=True)
    ab=np.array([r['is_antibody']=='True' for r in test])
    ax.scatter(y[~ab],pred[~ab],s=18,alpha=.5,color='#2D658A',label='Other PPI records')
    ax.scatter(y[ab],pred[ab],s=38,alpha=.9,edgecolor='#222222',linewidth=.4,color='#D58A36',label='Antibody-tagged diagnostic')
    lo,hi=min(y.min(),pred.min())-.4,max(y.max(),pred.max())+.4
    ax.plot([lo,hi],[lo,hi],ls='--',color='#777777',lw=1); ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Experimental mutation delta-delta-G (kcal/mol)',ylabel='MLP ensemble prediction (kcal/mol)',title=f"Held-out groups | n={len(test)} | Spearman={metrics['test']['mlp_ensemble']['spearman']:.3f}")
    ax.legend(frameon=False,fontsize=8); ax.grid(alpha=.15)
    fig.savefig(figures/'heldout_predictions.png',dpi=180); plt.close(fig)
    names=['mean','ridge','mlp_ensemble']; labels=['Training mean','Ridge regression','PyTorch MLP ensemble']
    values=np.array([metrics['test'][n]['rmse'] for n in names])
    intervals=np.array([metrics['test'][n]['cluster_bootstrap']['rmse_95_percent_interval'] for n in names])
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    ax.bar(np.arange(3),values,color=['#A6A6A6','#6A8D73','#2D658A'],width=.55)
    ax.errorbar(np.arange(3),values,yerr=np.vstack([values-intervals[:,0],intervals[:,1]-values]),fmt='none',ecolor='#333333',capsize=5)
    for i,v in enumerate(values): ax.text(i,v+.07,f'{v:.3f}',ha='center',va='bottom')
    ax.set_xticks(np.arange(3),labels); ax.set_ylabel('Test RMSE (kcal/mol), lower is better')
    ax.set_title('95% grouped-resampling intervals are wide (11 test groups)')
    ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
    fig.savefig(figures/'baseline_comparison.png',dpi=180); plt.close(fig)
    lines=['# PyTorch 蛋白互作突变效应预测 实际运行报告','',
           '本项目完成公开实验数据清洗、分组拆分、PyTorch 网络训练、验证集早停、独立测试、基线比较和单条推理。模型属于探索性基准，不是可直接部署的抗体亲和力优化工具。','',
           '## 数据与评估设计','',
           f"原始 SKEMPI 2.0 有 {audit['raw_rows']} 条记录；仅保留精确数值的亲和力和温度，排除截断值、缺失值及标记 assumed 的温度。相同复合物、突变和温度的重复测量以 ΔΔG 中位数汇总，保留 {audit['retained_unique']} 条记录。",'',
           f"训练 / 验证 / 测试为 {audit['split_counts']['train']} / {audit['split_counts']['validation']} / {audit['split_counts']['test']}，关联组数量为 {audit['split_group_counts']['train']} / {audit['split_group_counts']['validation']} / {audit['split_group_counts']['test']}。同一 PDB、显式 hold-out PDB 链接和可识别的完全相同蛋白名称连接到同组，组间不重叠；按是否含抗体标签分层。",'',
           '这不是完整序列聚类的同源隔离。没有把 AB/AG 等宽泛类别合并成一个组，因此也不是“整个抗体类别完全留出”的评估。名称别名、未标注同源体和实验异质性仍可能造成偏差。','',
           '标签：ΔΔG = RT ln(Kd_mut / Kd_wt)，单位 kcal/mol。负值表示该实验条件下突变后结合更紧，正值表示变弱；不预测绝对 Kd。','',
           '## 模型与训练','',
           f"输入为 451 维：原始 / 突变氨基酸计数、400 种替换类别、结构位置标签、相互作用伙伴侧别、突变数量及温度。结构位置标签来自数据集的原复合物注释，不是由本模型预测。网络 {config['model']}，共 {config['trainable_parameters']} 个可训练参数。",'',
           '使用 TensorDataset / DataLoader、nn.Module、MSELoss、反向传播、梯度裁剪和 AdamW。输入变换及标签均值方差仅在训练集拟合。岭回归正则化强度在验证集选择；神经网络按验证集早停，三个固定随机种子均纳入主集成模型，没有按测试集挑种子。','',
           f"Torch {config['torch_version']}；CPU {config['threads']} 线程；本次训练和评估约 {config['elapsed_seconds']:.1f} 秒。随机种子为 11、22、33。",'',
           '## 留出测试集结果','',
           '| 模型 | RMSE kcal/mol | MAE kcal/mol | Pearson | Spearman | R² |','|---|---:|---:|---:|---:|---:|']
    def fmt(v): return '不适用' if v is None else f'{v:.3f}'
    for n,label in zip(names,labels):
        m=metrics['test'][n]; lines.append('| '+label+' | '+' | '.join(fmt(m[k]) for k in ['rmse','mae','pearson','spearman','r2'])+' |')
    lines += ['', '神经网络的 RMSE 与岭回归相当，没有全面超过简单模型。R² 仅约 0.046，说明未见复合物上的数值解释能力有限；秩相关较高不能抵消大误差。测试仅 11 个关联组，按组自助采样的区间很宽，不能声称小幅差异具有统计学或商业意义。','',
              '## 抗体相关子集','',
              f"测试集中 AB/AG 标记记录只有 {audit['antibody_counts']['test']} 条。MLP RMSE={metrics['antibody_test_diagnostic']['mlp_ensemble']['rmse']:.3f}，Spearman={metrics['antibody_test_diagnostic']['mlp_ensemble']['spearman']:.3f}；该子集表现较弱，仅作诊断，不支持通用抗体优化能力的结论。",'',
              '## 已完成的检查','',
              '6 项单元检查覆盖标签符号和单位、截断数值排除、重复位点、特征顺序、关系分组和组间无交叉。训练过程检查 loss 有限，并保留权重、最佳轮次、所有逐条预测与分组指标。单条推理接口另行测试。','',
              '## 局限和下一步','',
              '模型没有抗原序列、局部三维几何、ESM 表征或抗体专用编码，不能将任意新肽序列输入后直接得到亲和力。多突变按计数聚合，不能充分表达空间协同作用。测试集现已用于报告，今后的模型开发需要新的独立测试或嵌套交叉验证，不能反复据此调参再称独立测试。','',
              '下一步可增加明确的序列同源分组、跨组交叉验证、抗体专用数据集和结构上下文；先在验证数据上确定方法，再保留新的独立评估。','',
              '## 简历可用表述','',
              '基于 SKEMPI 2.0 构建 PyTorch 蛋白互作突变 ΔΔG 回归项目，完成 3,998 条记录的清洗、关系分组留出、三随机种子训练、基线比较和结果追溯；在 478 条测试记录上获得 Spearman 0.549、RMSE 2.478 kcal/mol，并分析抗体子集及跨组泛化局限。项目由 AI 辅助完成，独立掌握程度需通过复现和代码解释验证。','',
              '数据来源与许可见项目 DATA_SOURCE.md。原数据按其 CC BY 4.0 条件保留署名，本项目清洗和建模不代表数据作者背书。']
    text='\n'.join(lines)+'\n'
    (run/'REPORT.md').write_text(text,encoding='utf-8')
    rows_html=''.join('<tr><td>'+html.escape(label)+'</td>'+''.join('<td>'+fmt(metrics['test'][n][k])+'</td>' for k in ['rmse','mae','spearman','r2'])+'</tr>' for n,label in zip(names,labels))
    images=''.join('<img alt="'+fn+'" src="data:image/png;base64,'+base64.b64encode((figures/fn).read_bytes()).decode()+'">' for fn in ['learning_curves.png','heldout_predictions.png','baseline_comparison.png'])
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PyTorch 突变效应预测报告</title><style>body{max-width:1050px;margin:36px auto;padding:0 24px;font:16px/1.7 sans-serif;color:#17212b}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #ddd;text-align:left}th{background:#eef2f5}img{display:block;max-width:100%;margin:30px auto}pre{white-space:pre-wrap;font:15px/1.75 sans-serif}</style><h1>PyTorch 蛋白互作突变效应预测</h1><p>公开实验数据 · 分组留出 · 三随机种子 · 完整基线与失败分析</p><table><tr><th>模型</th><th>RMSE</th><th>MAE</th><th>Spearman</th><th>R²</th></tr>'+rows_html+'</table>'+images+'<h2>完整报告</h2><pre>'+html.escape(text)+'</pre></html>'
    (run/'REPORT.html').write_text(page,encoding='utf-8')
    print('Created report, standalone HTML, and three scientific plots')
if __name__=='__main__': main()
