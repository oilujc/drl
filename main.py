from game import Game

from dqn import DQNAgent

def main():
    game = Game()

    state = game.get_state()
    agent = DQNAgent(state.shape[0], game.player.PLAYER_NUM_ACTIONS)

    while game.current_total_player_games < 100 and game.running == True:

        action = agent.get_action(state)
        game.update(action= action)

        _, action, reward, next_state, done = game.step(action, state)

        agent.push(state, action, reward, next_state, done)
        
        if len(agent.memory) > agent.batch_size:
            agent.update()
        
        game.render(game.screen)

        state = next_state

        if game.player.current_state == game.player.PLAYER_GET_PLAYER_REWARD_STATE:
            game.reset()


    game.quit()
    agent.save(game.current_total_player_games)
    
            
if __name__ == '__main__':
    main()

