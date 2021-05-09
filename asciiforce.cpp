// Script for use with dustmod.
// Prints an ascii display of the given width and height around the camera to the console.
// Charming.

const int WIDTH = 30;
const int HEIGHT = 20;
const bool SHOWPLAYER = false;
const int TIMEOUT_FRAMES = 300;

class script {
  camera@ cam;
  scene@ g;
  controllable@ player;
  array<int> rewards;
  array<int> dust;
  array<int> enemy_hp;
  array<int> filth_blocks;  
  array<int> breaks;
  int place;
  uint64 frame;
  uint32 sinceLastReward;

  script() {
    @cam = get_active_camera();
    @g = get_scene();
    for(int i = 0; i < 4; i++) {
      rewards.insertLast(0);
      dust.insertLast(0);
      enemy_hp.insertLast(0);
      filth_blocks.insertLast(0);
      breaks.insertLast(0);
    }
    place = 0;
    sinceLastReward = 0;
  }

  uint8 side_type(uint8 side) {
    if(side == 0) return ' '[0];
    else if(side >= 1 and side <= 5) return '2'[0];
    else return '3'[0];
  }

  void on_level_start() {
    frame = 0;
  }

  void on_level_end() {
    rewards[place] += 100;
  }

  void step(int entities) {
    frame++;
    int lastPlace = place;
    place = (place + 1) % 4;
    rewards[place] = 0;
    int filthblocks = 0;
    int cur_dust = 0;
    int cur_filth = 0;
    int cur_hp = 0;
    g.get_filth_remaining(cur_dust,
                          cur_filth,
                          cur_hp);
    dust[place] = cur_dust;
    filth_blocks[place] = cur_filth;
    enemy_hp[place] = cur_hp;
    breaks[place] = g.combo_break_count();
    rewards[place] = (((dust[lastPlace] - dust[place]) + (filth_blocks[lastPlace] - filth_blocks[place])) * 2 +
                      (enemy_hp[lastPlace] - enemy_hp[place]) * 4 -
                      (breaks[place] - breaks[lastPlace]) * 30);
    if(rewards[place] == 0) sinceLastReward++;
    else sinceLastReward = 0;
    int cur_reward = 0;
    for(int i = 0; i < 4; i++) cur_reward += rewards[i];
                      
    array<string> tiles;
    tileinfo@ tile;
    tilefilth@ filth;
    for(int i = 0; i < HEIGHT; i++) {
      tiles.insertLast("");
        for(int j = 0; j < WIDTH; j++) {
          int x = int((cam.x() / 48) + j - int(WIDTH / 2));
          int y = int((cam.y() / 48) + i - int(HEIGHT / 2));
          @tile = g.get_tile(x, y);
          tiles[i] += '0';
          if(tile.solid()) tiles[i][j] = '1'[0];
          if(tile.is_dustblock()) tiles[i][j] = '2'[0];
      }
    }
    for(int i = 0; i < HEIGHT; i++) {
        for(int j = 0; j < WIDTH; j++) {
          int x = int((cam.x() / 48) + j - int(WIDTH / 2));
          int y = int((cam.y() / 48) + i - int(HEIGHT / 2));
          @filth = g.get_tile_filth(x, y);
          uint8 top = side_type(filth.top());
          uint8 bottom = side_type(filth.bottom());
          uint8 left = side_type(filth.left());
          uint8 right = side_type(filth.right());
          if(top != ' '[0] and i > 0) tiles[i-1][j] = top;
          if(bottom != ' '[0] and i < 19) tiles[i+1][j] = bottom;
          if(left != ' '[0] and j > 0) tiles[i][j-1] = left;
          if(right != ' '[0] and j < 29) tiles[i][j+1] = right;
      }
    }
    int enemyCount = g.get_entity_collision(cam.y() - ((HEIGHT / 2) * 48),
                                            cam.y() + ((HEIGHT / 2) * 48),
                                            cam.x() - ((WIDTH / 2) * 48),
                                            cam.x() + ((WIDTH / 2) * 48),
                                            1);
    entity@ enemy;
    for(int i = 0; i < enemyCount; i++) {
      @enemy = g.get_entity_collision_index(i);
      controllable@ con = enemy.as_controllable();
      if(con.life() > 0) {
        int y = int(((enemy.y() - cam.y()) / 48) + (HEIGHT / 2) - 1);
        int x = int(((enemy.x() - cam.x()) / 48) + (WIDTH / 2) - 1);
        if(x >= 0 and x < WIDTH and y >= 0 and y < HEIGHT) tiles[y][x] = '4'[0];
      }
    }
    @player = controller_controllable(0);
    if(@player != null and SHOWPLAYER) {
      int y = int(((player.y() - cam.y()) / 48) + (HEIGHT / 2));
      int x = int(((player.x() - cam.x()) / 48) + (WIDTH / 2));
      tiles[y-1][x] = 'P'[0];
    }

    for(uint i = 0; i < tiles.length(); i++) puts("}" + tiles[i]);
    puts(")" + formatInt(cur_reward));
    puts("]" + formatInt(frame));
    if(sinceLastReward > TIMEOUT_FRAMES) puts("`");
    puts(">~>");    
  }
}
