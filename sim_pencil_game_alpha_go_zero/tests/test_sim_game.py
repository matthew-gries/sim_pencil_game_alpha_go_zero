from sim_pencil_game.SimGame import SimGame
import numpy as np


def test_move():

    sg = SimGame()

    for i in range(6):
        assert not sg.draw_line((i, i), 1)
        assert not sg.draw_line((i, i), -1)

    assert not sg.draw_line((-1, 0), 1)
    assert not sg.draw_line((6, 0), 1)
    assert not sg.draw_line((0, -1), 1)
    assert not sg.draw_line((0, 6), 1)
    assert not sg.draw_line((-1, 0), -1)
    assert not sg.draw_line((6, 0), -1)
    assert not sg.draw_line((0, -1), -1)
    assert not sg.draw_line((0, 6), -1)

    assert sg.draw_line((0, 1), 1)

    tmp = np.zeros((6, 6))
    tmp[0][1] = 1
    tmp[1][0] = 1

    assert np.array_equal(tmp, sg.adj)

    assert sg.draw_line((4, 5), -1)

    tmp[4][5] = -1
    tmp[5][4] = -1

    assert np.array_equal(tmp, sg.adj)

    assert sg.draw_line((3, 4), 1)
    assert not sg.draw_line((3, 4), 1)
    assert not sg.draw_line((3, 4), -1)
    assert not sg.draw_line((4, 3), 1)
    assert not sg.draw_line((4, 3), -1)


def test_did_player_win():

    sg = SimGame()

    assert not sg.did_this_player_win(1)
    assert not sg.did_this_player_win(-1)

    sg = SimGame()
    sg.draw_line((0, 1), 1)
    sg.draw_line((1, 2), 1)
    sg.draw_line((0, 2), 1)

    assert sg.did_this_player_win(-1)
    assert not sg.did_this_player_win(1)

    sg = SimGame()
    sg.draw_line((0, 1), -1)
    sg.draw_line((1, 2), -1)
    sg.draw_line((0, 2), -1)

    assert sg.did_this_player_win(1)
    assert not sg.did_this_player_win(-1)

    sg = SimGame()
    sg.draw_line((0, 1), -1)
    sg.draw_line((1, 2), 1)
    sg.draw_line((0, 2), -1)

    assert not sg.did_this_player_win(1)
    assert not sg.did_this_player_win(-1)

    sg = SimGame()
    sg.draw_line((0, 2), -1)
    sg.draw_line((2, 4), -1)
    sg.draw_line((0, 4), -1)

    assert sg.did_this_player_win(1)
    assert not sg.did_this_player_win(-1)


def test_get_legal_moves():

    sg = SimGame()

    all_moves = sg.get_legal_moves(1)
    assert all_moves == sg.get_legal_moves(-1)

    tmp = []
    for i in range(6):
        for j in range(6):
            if i != j:
                tmp.append((i, j))

    assert all_moves == set(tmp)

    sg.draw_line((0, 1), 1)

    tmp.remove((0, 1))
    tmp.remove((1, 0))

    assert sg.get_legal_moves(1) == set(tmp)
    assert sg.get_legal_moves(-1) == set(tmp)