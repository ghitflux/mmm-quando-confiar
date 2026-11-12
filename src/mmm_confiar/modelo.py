"""O modelo de mix de marketing (MMM), ajustado por máxima verossimilhança penalizada com restrições.

    vendas = base + soma_c  alfa_c * h( adstock(gasto_c / média_c ; retencao_c) ; K_c ) + erro

- base: constante, tendência, dois pares de seno e cosseno (sazonalidade anual e semestral) e a promoção;
- h: Hill com expoente 1, normalizada para valer 1 no gasto médio. Assim alfa_c é a venda semanal do canal no gasto
  médio, e K_c (em múltiplos do gasto médio) diz quão curva é a resposta;
- restrições: alfa_c >= 0 (anúncio não derruba venda), retenção entre 0 e 0,9, K entre 0,2 e 5;
- penalidade (o papel que as distribuições a priori têm num MMM bayesiano): a retenção é puxada para 0,3 e o
  logaritmo de K para 0 (meia saturação no gasto médio), com larguras de 0,3 e 0,7. São valores genéricos, iguais
  para todos os canais; sem eles, a forma da curva fica solta e a contribuição estimada varia sem controle.

Como os parâmetros são guardados. O que os dados medem bem é a inclinação de cada canal perto do gasto médio (quanto
a venda muda quando o gasto muda). A contribuição total exige extrapolar a curva até o gasto zero, e isso depende
de K. Por isso o ajuste trabalha com a inclinação s_c no lugar de alfa_c:

    alfa_c * h(a) = alfa_c + s_c * g(a ; K_c),   g(a ; K) = (1 + K)(a - 1) / (a + K),   alfa_c = s_c (1 + K_c) / K_c

g vale 0 e tem inclinação 1 no gasto médio; a parcela constante alfa_c vai para a constante da base. É o mesmo
modelo, escrito de modo que cada parâmetro mexa numa coisa só, o que faz o otimizador convergir em poucos passos.
Vetor de parâmetros: [base (q) | inclinações (C) | retenções (C) | log K (C)], tudo com vendas divididas pela média.

Com erro normal, maximizar a verossimilhança penalizada é minimizar uma soma de quadrados: os resíduos das vendas
mais um resíduo por penalidade, com peso sigma / largura. O sigma usado no peso é o desvio dos resíduos de uma
regressão linear simples (um piloto), para não depender do próprio ajuste.

O ajuste é um Levenberg-Marquardt com conjunto ativo para as restrições, escrito em NumPy e vetorizado: várias
séries (os pontos de partida, ou as réplicas do bootstrap) são ajustadas ao mesmo tempo, cada uma numa linha.
A incerteza vem de um bootstrap em blocos dos resíduos, que preserva a dependência entre semanas vizinhas.

Nada aqui usa bibliotecas de MMM. `tests/test_modelo.py` confere o jacobiano contra diferenças finitas e o ajuste
contra `scipy.optimize.least_squares`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mmm_confiar import transformacoes as tr

RETENCAO_MAXIMA = 0.9
K_MINIMO, K_MAXIMO = 0.2, 5.0
RETENCAO_CENTRAL, RETENCAO_LARGURA = 0.3, 0.3
LOGK_CENTRAL, LOGK_LARGURA = 0.0, 0.7
# Penalidade opcional no ROI (análise de sensibilidade): log ROI puxado para 0,2 com largura 0,9, os valores da
# distribuição a priori padrão do Meridian (Google) para o ROI, LogNormal(0,2; 0,9). Igual para todos os canais.
ROI_LOG_CENTRAL, ROI_LOG_LARGURA = 0.2, 0.9


@dataclass
class Desenho:
    """Os dados no formato que o ajuste usa (vendas e gastos em escala relativa, para o problema ficar bem condicionado)."""

    vendas: np.ndarray        # (T,) na unidade original
    escala: float             # venda média: y = vendas / escala
    Z: np.ndarray             # (T, q) regressoras da linha de base; a primeira coluna é a constante
    defasagens: np.ndarray    # (C, J, T) gasto relativo (gasto / média do canal): a linha k é o gasto de k semanas antes
    gasto_medio: np.ndarray   # (C,)
    gasto_total: np.ndarray   # (C,)

    @property
    def y(self) -> np.ndarray:
        return self.vendas / self.escala

    @property
    def semanas(self) -> int:
        return len(self.vendas)

    @property
    def q(self) -> int:
        return self.Z.shape[1]

    @property
    def canais(self) -> int:
        return self.defasagens.shape[0]

    @property
    def parametros(self) -> int:
        return self.q + 3 * self.canais

    @property
    def gasto_relativo(self) -> np.ndarray:
        """Gasto da própria semana, em múltiplos da média do canal: (T, C)."""
        return self.defasagens[:, 0, :].T


@dataclass
class Calibracao:
    """Resultado de experimentos usado para calibrar o modelo.

    Cada experimento mediu, com erro, a venda semanal média que um canal gera. A medida entra no ajuste como mais
    um resíduo: (medida - o que o modelo diz) / erro-padrão da medida, na mesma escala dos resíduos das vendas.
    """

    canais: np.ndarray        # índices dos canais testados
    medida: np.ndarray        # venda semanal média gerada pelo canal, medida pelo experimento (unidade original)
    erro_padrao: np.ndarray   # erro-padrão de cada medida (unidade original)


@dataclass
class Problema:
    """Tudo o que define a soma de quadrados a minimizar: os dados, o peso das penalidades e a calibração."""

    d: Desenho
    desvio_piloto: float                  # sigma usado para pôr penalidades e calibração na escala dos resíduos
    penalizar: bool = True                # penalidade na forma da curva (retenção e log K)
    calibracao: Calibracao | None = None
    penalizar_roi: bool = False           # penalidade genérica no ROI de cada canal (análise de sensibilidade)

    @property
    def linhas(self) -> int:
        extras = (2 * self.d.canais if self.penalizar else 0) + (self.d.canais if self.penalizar_roi else 0)
        return self.d.semanas + extras + (0 if self.calibracao is None else len(self.calibracao.canais))

    @property
    def piso_da_inclinacao(self) -> float:
        """Com penalidade no logaritmo do ROI, a inclinação não pode ser exatamente zero."""
        return 1e-5 if self.penalizar_roi else 0.0


@dataclass
class Ajuste:
    theta: np.ndarray               # (p,) melhor conjunto de parâmetros
    custo: float                    # soma dos quadrados (resíduos das vendas mais penalidades), escala relativa
    desvio_residual: float          # desvio-padrão dos resíduos das vendas, corrigido pelos graus de liberdade
    custos_das_partidas: np.ndarray
    problema: Problema

    @property
    def partidas_no_mesmo_otimo(self) -> float:
        """Parte dos pontos de partida que chegou ao mesmo custo do melhor (diagnóstico de mínimos locais)."""
        return float(np.mean(self.custos_das_partidas <= self.custo * (1 + 1e-4) + 1e-12))


def regressoras_da_base(semanas: int, promocao: np.ndarray) -> np.ndarray:
    """Constante, tendência centrada, senos e cossenos (anual e semestral) e a promoção.

    Com seno e cosseno juntos, a fase da sazonalidade é estimada: não é preciso saber em que semana do ano a série começa.
    """
    t = np.arange(semanas)
    angulo = 2 * np.pi * t / 52
    return np.column_stack([
        np.ones(semanas), (t - t.mean()) / 52, np.sin(angulo), np.cos(angulo), np.sin(2 * angulo), np.cos(2 * angulo), np.asarray(promocao, dtype=float),
    ])


def montar_desenho(tabela: pd.DataFrame, canais: tuple[str, ...]) -> Desenho:
    """Prepara os dados que o analista vê. O gasto anterior ao começo da série é suposto igual à média do canal."""
    gasto = tabela[[f"gasto_{c}" for c in canais]].to_numpy(dtype=float)
    media = gasto.mean(axis=0)
    defasagens = np.stack([tr.matriz_de_defasagens(gasto[:, c] / media[c], anterior=1.0).T for c in range(len(canais))])
    vendas = tabela["vendas"].to_numpy(dtype=float)
    return Desenho(
        vendas=vendas, escala=float(vendas.mean()), Z=regressoras_da_base(len(vendas), tabela["promocao"].to_numpy()),
        defasagens=np.ascontiguousarray(defasagens), gasto_medio=media, gasto_total=gasto.sum(axis=0),
    )


def desvio_piloto(d: Desenho) -> float:
    """Desvio dos resíduos de uma regressão linear das vendas na base e no gasto da semana (escala relativa)."""
    x = np.column_stack([d.Z, d.gasto_relativo])
    residuos = d.y - x @ np.linalg.lstsq(x, d.y, rcond=None)[0]
    return float(np.sqrt(residuos @ residuos / max(d.semanas - x.shape[1], 1)))


def limites(d: Desenho, piso_da_inclinacao: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    q, c = d.q, d.canais
    inferior = np.concatenate([np.full(q, -np.inf), np.full(c, piso_da_inclinacao), np.zeros(c), np.full(c, np.log(K_MINIMO))])
    superior = np.concatenate([np.full(q, np.inf), np.full(c, np.inf), np.full(c, RETENCAO_MAXIMA), np.full(c, np.log(K_MAXIMO))])
    return inferior, superior


def _adstock(pesos: np.ndarray, d: Desenho) -> np.ndarray:
    """Adstock de cada canal para cada linha de pesos: (B, C, J) -> (B, C, T)."""
    saida = np.empty((pesos.shape[0], d.canais, d.semanas))
    for canal in range(d.canais):
        saida[:, canal] = pesos[:, canal] @ d.defasagens[canal]
    return saida


def _curvas(theta: np.ndarray, d: Desenho, derivadas: bool = True):
    """Para cada linha de theta: g (B, C, T), a fração saturada a / (a + K), K e, se pedido, dg/dretencao e dg/dlogK."""
    q, c = d.q, d.canais
    retencao, k = theta[:, q + c: q + 2 * c], np.exp(theta[:, q + 2 * c:])[:, :, None]
    adstock = _adstock(tr.pesos_geometricos(retencao), d)
    inverso = 1.0 / (adstock + k)
    desvio = (adstock - 1.0) * inverso
    g = (1 + k) * desvio
    if not derivadas:
        return g, adstock * inverso, k
    d_adstock = _adstock(tr.derivada_dos_pesos_geometricos(retencao), d)
    dg_retencao = ((1 + k) * inverso) ** 2 * d_adstock
    dg_logk = k * desvio**2
    return g, adstock * inverso, k, dg_retencao, dg_logk, inverso, d_adstock


def prever(theta: np.ndarray, d: Desenho) -> np.ndarray:
    """Vendas previstas (escala relativa) para cada linha de theta: (B, T)."""
    theta = np.atleast_2d(theta)
    g, _, _ = _curvas(theta, d, derivadas=False)
    return theta[:, : d.q] @ d.Z.T + np.einsum("bc,bct->bt", theta[:, d.q: d.q + d.canais], g)


def residuos_e_jacobiano(theta: np.ndarray, y: np.ndarray, problema: Problema, medida: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Resíduos (observado - previsto) e jacobiano da previsão, para cada linha de theta: (B, n) e (B, p, n).

    O jacobiano vem com os parâmetros na segunda dimensão (cada parâmetro é uma linha contígua na memória).
    As n observações são: as semanas; depois, se houver penalidade, uma por canal para a retenção e uma para log K;
    depois, se houver calibração, uma por experimento: peso * (medida - venda semanal média do canal no modelo).
    `medida` (B, experimentos), na unidade original, substitui a da calibração (o bootstrap sorteia uma por réplica).
    """
    d = problema.d
    q, c, t, lotes = d.q, d.canais, d.semanas, theta.shape[0]
    g, fracao, k, dg_retencao, dg_logk, inverso, d_adstock = _curvas(theta, d)
    s = theta[:, q: q + c, None]
    r = np.empty((lotes, problema.linhas))
    jac = np.zeros((lotes, d.parametros, problema.linhas))
    r[:, :t] = y - (theta[:, :q] @ d.Z.T + (s * g).sum(axis=1))
    jac[:, :q, :t] = d.Z.T
    jac[:, q: q + c, :t] = g
    jac[:, q + c: q + 2 * c, :t] = s * dg_retencao
    jac[:, q + 2 * c:, :t] = s * dg_logk
    coluna = t
    if problema.penalizar:
        canais = np.arange(c)
        peso_retencao, peso_logk = problema.desvio_piloto / RETENCAO_LARGURA, problema.desvio_piloto / LOGK_LARGURA
        r[:, coluna: coluna + c] = peso_retencao * (RETENCAO_CENTRAL - theta[:, q + c: q + 2 * c])
        jac[:, q + c + canais, coluna + canais] = peso_retencao
        r[:, coluna + c: coluna + 2 * c] = peso_logk * (LOGK_CENTRAL - theta[:, q + 2 * c:])
        jac[:, q + 2 * c + canais, coluna + c + canais] = peso_logk
        coluna += 2 * c
    if problema.penalizar_roi:
        # log ROI_c = log s_c + log((1 + K)^2 / K) + log(média de a / (a + K)) + log(escala * T / gasto total)
        canais = np.arange(c)
        peso = problema.desvio_piloto / ROI_LOG_LARGURA
        kc, sc = k[:, :, 0], np.maximum(theta[:, q: q + c], 1e-12)
        media_fracao = fracao.mean(axis=2)
        log_roi = np.log(sc) + np.log((1 + kc) ** 2 / kc) + np.log(media_fracao) + np.log(d.escala * t / d.gasto_total)
        r[:, coluna: coluna + c] = peso * (ROI_LOG_CENTRAL - log_roi)
        jac[:, q + canais, coluna + canais] = peso / sc
        jac[:, q + c + canais, coluna + canais] = peso * (k * inverso**2 * d_adstock).mean(axis=2) / media_fracao
        jac[:, q + 2 * c + canais, coluna + canais] = peso * ((kc - 1) / (kc + 1) - kc * (fracao * inverso).mean(axis=2) / media_fracao)
        coluna += c
    if problema.calibracao is not None:
        cal = problema.calibracao
        medida = cal.medida[None, :] if medida is None else medida
        for i, canal in enumerate(cal.canais):
            # venda semanal média do canal no modelo: s (1 + K)^2 / K * média de a / (a + K)
            peso = problema.desvio_piloto / (cal.erro_padrao[i] / d.escala)
            kc, sc = k[:, canal, 0], theta[:, q + canal]
            fator = (1 + kc) ** 2 / kc
            media_fracao = fracao[:, canal].mean(axis=1)
            r[:, coluna + i] = peso * (medida[:, i] / d.escala - sc * fator * media_fracao)
            jac[:, q + canal, coluna + i] = peso * fator * media_fracao
            jac[:, q + c + canal, coluna + i] = peso * sc * fator * kc * (inverso[:, canal] ** 2 * d_adstock[:, canal]).mean(axis=1)
            d_media_fracao = -(fracao[:, canal] * inverso[:, canal]).mean(axis=1)  # derivada em relação a K
            jac[:, q + 2 * c + canal, coluna + i] = peso * sc * kc * ((kc**2 - 1) / kc**2 * media_fracao + fator * d_media_fracao)
    return r, jac


def levenberg_marquardt(
    theta: np.ndarray, y: np.ndarray, problema: Problema, medida: np.ndarray | None = None,
    iteracoes: int = 100, tolerancia: float = 1e-9, amortecimento: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimiza a soma dos quadrados de cada linha de theta, respeitando os limites. Devolve (theta, custo).

    A cada passo resolve (JJ' + mu * diag(JJ')) delta = J r só nos parâmetros livres. Um parâmetro encostado num
    limite com o gradiente empurrando para fora fica travado naquele passo (conjunto ativo). O passo é aceito se
    o custo cai; mu diminui quando aceita e aumenta quando rejeita. Linhas que já convergiram saem da conta.
    """
    inferior, superior = limites(problema.d, problema.piso_da_inclinacao)
    theta = np.clip(np.array(theta, dtype=float), inferior, superior)
    lotes, p = theta.shape
    y = np.broadcast_to(y, (lotes, problema.d.semanas))
    if medida is not None:
        medida = np.broadcast_to(medida, (lotes, medida.shape[-1]))
    r, jac = residuos_e_jacobiano(theta, y, problema, medida)
    custo = np.einsum("bn,bn->b", r, r)
    mu = np.full(lotes, amortecimento)
    vivos = np.arange(lotes)  # linhas que ainda não convergiram; r e jac guardam só essas
    diagonal = np.arange(p)
    for _ in range(iteracoes):
        atual = theta[vivos]
        gradiente = (jac @ r[:, :, None])[:, :, 0]  # direção de descida: aumentar o parâmetro reduz o custo se > 0
        livre = ~(((atual <= inferior + 1e-12) & (gradiente < 0)) | ((atual >= superior - 1e-12) & (gradiente > 0)))
        hessiana = jac @ jac.transpose(0, 2, 1)
        escala = np.maximum(hessiana[:, diagonal, diagonal], 1e-12)
        hessiana *= livre[:, :, None] & livre[:, None, :]
        hessiana[:, diagonal, diagonal] = np.where(livre, escala * (1 + mu[vivos, None]), 1.0)
        passo = np.linalg.solve(hessiana, (gradiente * livre)[:, :, None])[:, :, 0]
        candidato = np.clip(atual + passo, inferior, superior)
        r_novo, jac_novo = residuos_e_jacobiano(candidato, y[vivos], problema, None if medida is None else medida[vivos])
        custo_novo = np.einsum("bn,bn->b", r_novo, r_novo)
        aceito = custo_novo < custo[vivos]
        melhora = np.where(aceito, custo[vivos] - custo_novo, np.inf)
        if not aceito.all():  # as linhas rejeitadas ficam onde estavam
            rejeitado = ~aceito
            r_novo[rejeitado], jac_novo[rejeitado] = r[rejeitado], jac[rejeitado]
        r, jac = r_novo, jac_novo
        theta[vivos[aceito]] = candidato[aceito]
        custo[vivos[aceito]] = custo_novo[aceito]
        mu[vivos] = np.where(aceito, np.maximum(mu[vivos] / 3, 1e-9), mu[vivos] * 4)
        terminou = (melhora <= tolerancia * custo[vivos]) | (mu[vivos] > 1e9)
        if terminou.any():
            vivos, r, jac = vivos[~terminou], r[~terminou], jac[~terminou]
            if len(vivos) == 0:
                break
    return theta, custo


def _partida_linear(retencao: np.ndarray, logk: np.ndarray, d: Desenho) -> np.ndarray:
    """Dadas a retenção e a saturação, a parte linear (base e inclinações) sai por mínimos quadrados; inclinação negativa vira zero."""
    lotes, q, c = retencao.shape[0], d.q, d.canais
    theta = np.zeros((lotes, d.parametros))
    theta[:, q + c: q + 2 * c], theta[:, q + 2 * c:] = retencao, logk
    g, _, _ = _curvas(theta, d, derivadas=False)
    x_t = np.concatenate([np.broadcast_to(d.Z.T, (lotes, q, d.semanas)), g], axis=1)
    linear = np.linalg.solve(x_t @ x_t.transpose(0, 2, 1) + 1e-8 * np.eye(q + c), x_t @ d.y[:, None])[:, :, 0]
    theta[:, :q], theta[:, q: q + c] = linear[:, :q], np.maximum(linear[:, q:], 0.0)
    return theta


def ajustar(
    d: Desenho, partidas: int = 8, semente: int = 0, calibracao: Calibracao | None = None, penalizar: bool = True, penalizar_roi: bool = False
) -> Ajuste:
    """Ajusta o MMM a partir de vários pontos de partida e fica com o de menor custo."""
    gerador = np.random.default_rng([semente, 505])
    retencao = np.vstack([np.full(d.canais, RETENCAO_CENTRAL), gerador.uniform(0.0, 0.8, (partidas - 1, d.canais))])
    logk = np.vstack([np.zeros(d.canais), gerador.uniform(np.log(0.3), np.log(3.0), (partidas - 1, d.canais))])
    problema = Problema(d=d, desvio_piloto=desvio_piloto(d), penalizar=penalizar, calibracao=calibracao, penalizar_roi=penalizar_roi)
    thetas, custos = levenberg_marquardt(_partida_linear(retencao, logk, d), d.y, problema)
    melhor = int(np.argmin(custos))
    residuos = d.y - prever(thetas[melhor], d)[0]
    desvio = float(np.sqrt(residuos @ residuos / max(d.semanas - d.parametros, 1)))
    return Ajuste(theta=thetas[melhor].copy(), custo=float(custos[melhor]), desvio_residual=desvio, custos_das_partidas=custos, problema=problema)


def tamanho_do_bloco(semanas: int) -> int:
    """Comprimento do bloco do bootstrap: cresce com a raiz cúbica do tamanho da série (6 semanas em 1 ano, 9 em 4)."""
    return max(4, int(round(1.5 * semanas ** (1 / 3))))


def indices_em_blocos(semanas: int, bloco: int, repeticoes: int, gerador: np.random.Generator) -> np.ndarray:
    """Índices de um bootstrap em blocos circulares: cada réplica é uma sequência de blocos de semanas vizinhas."""
    n_blocos = int(np.ceil(semanas / bloco))
    inicios = gerador.integers(0, semanas, size=(repeticoes, n_blocos))
    indices = (inicios[:, :, None] + np.arange(bloco)) % semanas
    return indices.reshape(repeticoes, -1)[:, :semanas]


def bootstrap_em_blocos(ajuste: Ajuste, repeticoes: int = 200, semente: int = 0, bloco: int | None = None) -> np.ndarray:
    """Réplicas dos parâmetros por bootstrap em blocos dos resíduos: (repeticoes, p).

    Cada réplica soma à previsão ajustada uma sequência de blocos de resíduos sorteados com reposição e reajusta o
    modelo partindo da estimativa. Os resíduos são centrados e inflados por sqrt(T / (T - p)), porque um modelo com
    muitos parâmetros deixa resíduos menores do que o erro verdadeiro. Com calibração, a medida do experimento
    também é sorteada de novo a cada réplica, com o erro-padrão dela.
    """
    d = ajuste.problema.d
    gerador = np.random.default_rng([semente, 606])
    bloco = bloco or tamanho_do_bloco(d.semanas)
    previsto = prever(ajuste.theta, d)[0]
    residuos = d.y - previsto
    residuos = (residuos - residuos.mean()) * np.sqrt(d.semanas / max(d.semanas - d.parametros, 1))
    y = previsto + residuos[indices_em_blocos(d.semanas, bloco, repeticoes, gerador)]
    medida = None
    if ajuste.problema.calibracao is not None:
        cal = ajuste.problema.calibracao
        medida = cal.medida + cal.erro_padrao * gerador.normal(size=(repeticoes, len(cal.canais)))
    thetas, _ = levenberg_marquardt(np.tile(ajuste.theta, (repeticoes, 1)), y, ajuste.problema, medida, iteracoes=40, amortecimento=1e-3)
    return thetas


def contribuicao_total(theta: np.ndarray, d: Desenho) -> np.ndarray:
    """Venda total atribuída a cada canal no período, na unidade original: (B, C).

    É a soma, nas semanas, de alfa_c * h(adstock): o que o modelo diz que se perderia zerando o canal.
    """
    theta = np.atleast_2d(theta)
    _, fracao, k = _curvas(theta, d, derivadas=False)
    k = k[:, :, 0]
    return d.escala * theta[:, d.q: d.q + d.canais] * (1 + k) ** 2 / k * fracao.sum(axis=2)


def roi(theta: np.ndarray, d: Desenho) -> np.ndarray:
    """Venda atribuída por real gasto, por canal: (B, C)."""
    return contribuicao_total(theta, d) / d.gasto_total


def parametros_por_canal(theta: np.ndarray, d: Desenho) -> dict[str, np.ndarray]:
    """Traduz o vetor de parâmetros para as grandezas do modelo, na unidade original."""
    q, c = d.q, d.canais
    k = np.exp(theta[q + 2 * c:])
    return {
        "venda_no_gasto_medio": d.escala * theta[q: q + c] * (1 + k) / k,
        "inclinacao_no_gasto_medio": d.escala * theta[q: q + c] / d.gasto_medio,  # venda a mais por real a mais
        "retencao": theta[q + c: q + 2 * c].copy(),
        "meia_saturacao": k,
    }


def resposta_por_canal(theta: np.ndarray, d: Desenho, gasto: np.ndarray) -> np.ndarray:
    """Curva de resposta estimada: venda semanal de cada canal com o gasto semanal `gasto` mantido constante."""
    partes = parametros_por_canal(theta, d)
    return partes["venda_no_gasto_medio"] * tr.hill_normalizada(np.asarray(gasto, dtype=float) / d.gasto_medio, partes["meia_saturacao"])


def intervalo(amostras: np.ndarray, nivel: float) -> tuple[np.ndarray, np.ndarray]:
    """Intervalo percentil das réplicas do bootstrap, coluna por coluna.

    Usa as posições k / (B + 1), que dão a cobertura certa em média mesmo com poucas réplicas.
    """
    cauda = (1 - nivel) / 2
    return np.quantile(amostras, cauda, axis=0, method="weibull"), np.quantile(amostras, 1 - cauda, axis=0, method="weibull")
