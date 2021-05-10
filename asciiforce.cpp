// Script for use with dustmod.
// Prints an ascii display of the given width and height around the camera to the console.
// Charming.

const int WIDTH = 32;
const int HEIGHT = 24;
const bool SHOWPLAYER = false;
const int TIMEOUT_FRAMES = 300;
const bool DEBUG = false;
const int IDLE_PUNISHMENT = 20;
const int IDLE_PUNISHMENT_INTERVAL = 30;

class script {
  camera@ cam;
  scene@ g;
  controllable@ player;
  array<int> dust;
  array<int> enemy_hp;
  array<int> filth_blocks;  
  int place;
  int last_place;
  uint64 frame;
  uint32 since_last_reward;
  int total_reward;

  void init_arrays() {
    for(int i = 0; i < 4; i++) {
      dust.insertLast(0);
      enemy_hp.insertLast(0);
      filth_blocks.insertLast(0);
    }
  }

  void reinit_arrays(int dust_c, int hp_c, int filthblock_c) {
    for(int i = 0; i < 4; i++) {
      dust[i] = dust_c;
      enemy_hp[i] = hp_c;
      filth_blocks[i] = filthblock_c;
    }
  }
  
  script() {
    @cam = get_active_camera();
    @g = get_scene();
    init_arrays();
    place = 0;
    since_last_reward = 0;
    total_reward = 0;
    last_place = 0;
  }

  uint8 side_type(uint8 side) {
    if(side == 0) return ' '[0];
    else if(side >= 1 and side <= 5) return '2'[0];
    else return '3'[0];
  }

  void on_level_start() {
    frame = 0;
    int cur_dust = 0;
    int cur_filth = 0;
    int cur_hp = 0;
    g.get_filth_remaining(cur_dust,
                          cur_filth,
                          cur_hp);
    reinit_arrays(cur_dust, cur_hp, cur_filth);
  }

  void on_level_end() {
    total_reward += 100;
  }

  void update_reward() {
    int cur_reward = 0;
    int cur_dust = 0;
    int cur_filthblocks = 0;
    int cur_hp = 0;
    g.get_filth_remaining(cur_dust,
                          cur_filthblocks,
                          cur_hp);
    dust[place] = cur_dust;
    filth_blocks[place] = cur_filthblocks;
    enemy_hp[place] = cur_hp;
    cur_reward += (((dust[last_place] - dust[place]) + (filth_blocks[last_place] - filth_blocks[place])) * 10 +
                      (enemy_hp[last_place] - enemy_hp[place]) * 40);
    if(cur_reward == 0) {
      since_last_reward++;
      if(since_last_reward % IDLE_PUNISHMENT_INTERVAL == 0) {
        cur_reward -= IDLE_PUNISHMENT;
      }
    } else {
      since_last_reward = 0;
    }
    total_reward += cur_reward;
  }

  array<string> get_screen() {
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
    return tiles;
  }

  void step(int entities) {
    last_place = place;
    place = (place + 1) % 4;
    frame++;
    update_reward();
    array<string> screen = get_screen();

    // printing section
    for(uint i = 0; i < screen.length(); i++) puts("}" + screen[i]);
    puts(")" + formatInt(total_reward));
    puts("]" + formatInt(frame));
    if(since_last_reward > TIMEOUT_FRAMES) puts("`");
    puts(">~>");
    if(DEBUG) {
      // outputs not prefaced are retransmitted after interception,
      // so these will print
      for(uint i = 0; i < screen.length(); i++) puts(screen[i]);
      puts(formatInt(total_reward));
      puts(formatInt(frame));
    }
  }
}
