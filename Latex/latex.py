from jinja2 import Environment, FileSystemLoader
from math import factorial


def matr_to_table_r_int(mat, delim):
    s = ""
    for i in mat:
        for x in i:
           s += f'{int(x)}'
           s += f'{delim}'
        s = s[:-len(delim)]
        s = s + " \\\\\n"
    s = s[:-3]
    return s


def vector_tex(vec, delim):
    s = ""

    for i in vec:
        s = s + str(i) + delim
    return s[:-(len(delim))]


def vector_tex_round(vec, delim, round_n):
    s = ""

    for i in vec:
        s = s + str(round(i, round_n)) + delim
    return s[:-(len(delim))]


def make_latex(var, g, name, name_short, lambda_A, lambda_B, lambda_S, Na, Nb, Ra, Rb, graph_tex, graph_dot_ls,
               matrix_np, kolmogorov, kolmogorov_0, p_pred, mu, kzrs, snge, t_sr_v, p_pred_exp,
               t_sr_v_dsm, p_pred_exp_dsm, modeling_log):
    # Jinja init
    environment = Environment(
        loader=FileSystemLoader("Latex/templates/")
    )

    # Preamble text
    base_template = environment.get_template("educmm_lab_Variant_N_M-id.tex")
    base_res_file_name = "Latex/res/labs/educmm_txb_COMPMATHLAB-Solution_N_M/educmm_lab_Variant_N_M-id.tex"
    base_text = base_template.render(
        author_name="{" + str(name) + "}",
        author_name_short="{" + str(name_short) + "}",
        group="{" + f"РК6-8{g}б" + "}",
        variant="{" + str(var) + "}"
    )

    with open(base_res_file_name, mode="w+", encoding="utf-8") as base:
        base.write(base_text)
        print(f"... wrote {base_res_file_name}")

    # Main text
    latex_text_template = environment.get_template("educmm_txb_COMPMATHLAB-Solution_N_M.tex")
    latex_text_file_name = f"Latex/res/labs/educmm_txb_COMPMATHLAB-Solution_N_M.tex"
    latex_text = latex_text_template.render(
        g=g,
        var=var,
        size=len(matrix_np),
        lambda_A=lambda_A,
        lambda_B=lambda_B,
        Na=Na,
        Nb=Nb,
        Ra=Ra,
        Rb=Rb,
        lambda_S=lambda_S,
        G=graph_tex,
        G_new=graph_dot_ls,
        mat=matr_to_table_r_int(matrix_np, ' & '),
        kolmogorov_0=kolmogorov_0,
        kolmogorov=kolmogorov,
        p_pred=vector_tex_round(p_pred, delim=", ", round_n=2),
        mu=round(mu, 6),
        kzrs=round(kzrs, 5),
        sngeA=round(snge[0], 2),
        sngeB=round(snge[1], 2),
        t_sr_v=t_sr_v,
        p_pred_exp=vector_tex_round(p_pred_exp, delim=", ", round_n=2),
        t_sr_v_dsm=t_sr_v_dsm,
        p_pred_exp_dsm=p_pred_exp_dsm,
        modeling_log=modeling_log
    )

    with open(latex_text_file_name, mode="w+", encoding="utf-8") as text:
        text.write(latex_text)
        print(f"... wrote {latex_text_file_name}")



