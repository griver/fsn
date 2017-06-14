# coding=utf-8
"""
В файле находятся функции переводяшие некоторые структуры данных связанные с графами в формат dot
и сохраняющие их в файле.
Данные в формате dot можно визуализировать при помощи утилиты graphviz
"""
from ..old_fslib import FSBuilder


def secondary_graph2dot(net, vf, filename=None):
    assert isinstance(net, FSBuilder.BaseFSNetwork)
    secs = net.all_secondary()

    verts = set(map(lambda x: x.IA_point(), secs))
    verts.update(map(lambda x: x.AR_point(), secs))

    result = "digraph " + net.get_name() + " {\n"

    for v in verts:
        result += "   v"+ str(v.get_id()) + " [label=\""+v.name + "\""
        if vf.has(v):
            result += ",fillcolor=\"palegreen\",style=\"filled\""
        result += "];\n"


    for s in secs:
        result += "   v" + str(s.IA_point().get_id()) + " -> v" + str(s.AR_point().get_id())
        if vf.has(s):
            if vf.get(s) > 0: result += "[color=\"green\", label=\"g("+str(vf.get(s)) + ")\"]"
            else: result += "[color=\"red\", label=\"b("+str(vf.get(s)) + ")\"]"
        result += "\n"
    result += "}"

    if filename is None: filename = net.get_name() + ".dot"
    f = open(filename, 'w')
    f.write(result)
    f.close()


def action_value_function2dot(net, vf, filename=None):
    assert isinstance(net, FSBuilder.BaseFSNetwork)
    secs = net.all_secondary()

    verts = set(map(lambda x: x.IA_point(), secs))
    verts.update(map(lambda x: x.AR_point(), secs))

    result = "digraph " + net.get_name() + " {\n"

    for v in verts:
        result += "   v"+ str(v.get_id()) + " [label=\""+v.name + "\""
        if vf.has(v):
            result += ",fillcolor=\"palegreen\",style=\"filled\""
        result += "];\n"


    for s in secs:
        result += "   v" + str(s.IA_point().get_id()) + " -> v" + str(s.AR_point().get_id())
        if vf.has(s):
            if vf.get(s) > 0: result += "[color=\"green\", label=\""+str(round(vf.get(s), 3)) + ")\"]"
            else: result += "[color=\"red\", label=\""+str(round(vf.get(s), 3)) + "\"]"
        result += "\n"
    result += "}"

    if filename is None: filename = net.get_name() + ".dot"
    f = open(filename, 'w')
    f.write(result)
    f.close()


def state_value_function2dot(net, vf, filename=None):
    assert isinstance(net, FSBuilder.BaseFSNetwork)
    secs = net.all_secondary()

    verts = set(map(lambda x: x.IA_point(), secs))
    verts.update(map(lambda x: x.AR_point(), secs))

    result = "digraph " + net.get_name() + " {\n"

    for v in verts:
        result += "   v"+ str(v.get_id()) + " [label=\""+v.name + " val: " + str(round(vf.get(v), 3)) + "\""
        if vf.has(v):
            result += ",fillcolor=\"palegreen\",style=\"filled\""
        result += "];\n"


    for s in secs:
        result += "   v" + str(s.IA_point().get_id()) + " -> v" + str(s.AR_point().get_id())
        if vf.has(s):
            if vf.get(s) > 0: result += "[label=\"" + str(vf.get(s)) + "\"]"
        result += "\n"
    result += "}"

    if filename is None: filename = net.get_name() + ".dot"
    f = open(filename, 'w')
    f.write(result)
    f.close()


def environment2dot(env, vf, filename=None):
    result = "digraph " + env._name + " {\n"

    for v in env.vertices():
        result += "   v"+ str(v.get_id()) + " [label=\""+v.name + "\""
        if vf.has(v):
            result += ",fillcolor=\"palegreen\",style=\"filled\""
        result += "];\n"

    for v in env.vertices():
        for e  in v.get_outcoming():
            result += "   v" + str(v.get_id()) + " -> v" + str(e.get_dst().get_id())

    result += "}"

    if filename is None: filename = env.get_name() + ".dot"
    f = open(filename, 'w')
    f.write(result)
    f.close()
    #subprocess.call("dot -Tpng  -o draw_env_" + env.get_name() + ".png" + filename, shell=True)


def stoch_environment2dot(env, filename=None):
    result = "digraph " + env._name + " {\n"

    for v in env.vertices():
        result += "   v"+ str(v.get_id()) + " [label=\""+str(v.get_id()) + "\""
        #if vf.has(v):
        #    result += ",fillcolor=\"palegreen\",style=\"filled\""
        result += "];\n"

    for v in env.vertices():

        for i in xrange(len(v.get_outcoming())):
            e = v.get_outcoming()[i]
            result += "   v" + str(v.get_id()) + " -> v" + str(e.get_dst().get_id())
            result += "[ label=\"" +str(i) + "\"," \
                      " color=" + ("\"black\"" if e.is_available() else "\"red\"") + "];\n"

    result += "}"

    if filename is None: filename = env.get_name() + ".dot"
    f = open(filename, 'w')
    f.write(result)
    f.close()