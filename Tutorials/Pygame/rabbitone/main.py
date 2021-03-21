# 1. 파이 게임 모듈을 불러온다.
import pygame
import math
import random
from queue import Queue

# root_dir = "pygame/rabbitone"
root_dir = "./rabbitone"


# 2. 초기화 시킨다.
# num_pass: 초기화가 성공한 모듈 갯수
# num_fail: 초기화가 실패한 모듈 갯수
num_pass, num_fail = pygame.init()

# 화면의 크기는 640x480
world_width, world_height = 640, 480
screen = pygame.display.set_mode((world_width, world_height))

# 키 정보 [W, A, S, D]
keys = [False, False, False, False]

# 플레이어 위치 정보
player_pos = [100, 100]

# 화살 정보
arrow_infos = Queue()

# 화살 명중률 [명중한 갯수, 발사한 갯수]
accuracy = [0, 0]

# 적들의 출현 시간
timer = 0
badguy_appear = 100

# 적들의 출현 위치
badguy_infos = Queue()
badguy_infos.put([640, 100])

# 플레이어 생명력
player_health = 194

# 제한된 플레이 시간(ms)
play_time = 90000


# 3. 이미지를 가져온다.
player = pygame.image.load(root_dir + "/resources/images/dude.png")
player_rot = None
grass = pygame.image.load(root_dir + "/resources/images/grass.png")
castle = pygame.image.load(root_dir + "/resources/images/castle.png")
arrow = pygame.image.load(root_dir + "/resources/images/bullet.png")
badguy = pygame.image.load(root_dir + "/resources/images/badguy.png")
healthbar = pygame.image.load(root_dir + "/resources/images/healthbar.png")
health = pygame.image.load(root_dir + "/resources/images/health.png")
gameover = pygame.image.load(root_dir + "/resources/images/gameover.png")
youwin = pygame.image.load(root_dir + "/resources/images/youwin.png")


# 3.1 오디오를 가져온다.
# 오디오를 사용하기 위한 초기화
pygame.mixer.init()
hit = pygame.mixer.Sound(root_dir + "/resources/audio/explode.wav")
enemy = pygame.mixer.Sound(root_dir + "/resources/audio/enemy.wav")
shoot = pygame.mixer.Sound(root_dir + "/resources/audio/shoot.wav")

# 볼륨 설정
hit.set_volume(0.05)
enemy.set_volume(0.05)
shoot.set_volume(0.05)

# 배경음악 설정
pygame.mixer.music.load(root_dir + "/resources/audio/moonlight.wav")
pygame.mixer.music.play(-1, 0.0)
pygame.mixer.music.set_volume(0.25)


# 4. 계속 화면이 보이도록 한다.
running = True
is_win = True

while running:
    timer += 1


    # 5. 화면을 깨끗하게 한다.
    screen.fill((0, 0, 0))  # (R, G, B)


    # 6. 모든 요소들을 다시 그린다.
    for x in range(world_width // grass.get_width() + 1):
        for y in range(world_height // grass.get_height() + 1):
            screen.blit(grass, (x * grass.get_width(), y * grass.get_height()))
    
    # castle 이미지를 (x, y) 위치에 그린다.
    screen.blit(castle, (0, 30))
    screen.blit(castle, (0, 135))
    screen.blit(castle, (0, 240))
    screen.blit(castle, (0, 345))


    # 6.1 플레이어의 회전 그리기 (위치와 각도 설정)
    mouse_pos = pygame.mouse.get_pos()
    # 플레이어의 위치부터 마우스 위치까지의 Θ값을 구한다.
    # 플레이어 크기가 64x46이므로, 플레이어의 중심으로부터 마우스 위치까지의 Θ값를 구하기 위해 보정하였다. (player_pos[1] + 32, player_pos[0] + 26)
    angle = math.atan2(mouse_pos[1] - (player_pos[1] + player.get_height() // 2), mouse_pos[0] - (player_pos[0] + player.get_width() // 2))
    # 이때 반환되는 angle은 라디안(radian)값을 갖는다. (1 radian = 180°/𝛑)
    # 라디안 값을 갖는 angle을 degree로 바꾸기 위해 57.29(180°/𝛑)을 곱한다.
    player_rot = pygame.transform.rotate(player, 360 - angle * (180 / math.pi))
    # angle을 구하기 위해 플레이어의 중심을 기준으로 계산한 것 같이,
    # 회전된 플레이어의 위치를 그려주기 위해 새롭게 회전된 플레이어의 중심을 기준으로 현재 플레이어의 위치를 .보정한다.
    player_rot_pos = (player_pos[0] + (player.get_width() - player_rot.get_width()) // 2, player_pos[1] + (player.get_height() - player_rot.get_height()) // 2)
    # # 보정이 완료된 플레이어를 그린다.
    screen.blit(player_rot, player_rot_pos)


    # 6.2 화살 그리기 
    index = 0
    for _ in range(arrow_infos.qsize()):
        arrow_angle, [arrow_x, arrow_y] = arrow_infos.get()
        
        # 화살의 속도 성분을 구하고 속력 10을 곱한다.
        velocity_x = math.cos(arrow_angle) * 10
        velocity_y = math.sin(arrow_angle) * 10
        # 화살의 좌표는 화살의 속도만큼 이동한다.
        arrow_x += velocity_x
        arrow_y += velocity_y

        # 새로 이동한 화살이 화면 밖을 벗어날 경우, 그리지 않는다.
        if arrow_x < -max(arrow.get_width(), arrow.get_height()) or world_width < arrow_x \
        or arrow_y < -max(arrow.get_width(), arrow.get_height()) or world_height < arrow_y:
            pass
        else:
            # 새로 이동한 화살이 화면 안에 있을 경우, 화살을 그린다.
            arrow_rot = pygame.transform.rotate(arrow, 360 - arrow_angle * 57.29)
            screen.blit(arrow_rot, (arrow_x, arrow_y))
            # 화살을 그린 뒤에, 다시 화살 정보에 추가한다.
            arrow_infos.put([arrow_angle, [arrow_x, arrow_y]])


    # 6.3 적들 그리기
    badguy_appear = random.randint(70, 100)
    if timer % badguy_appear == 0:
        badguy_infos.put([640, random.randint(50, 430)])

    for _ in range(badguy_infos.qsize()):
        badguy_x, badguy_y = badguy_infos.get()
        # 적들의 위치를 -7만큼 조절
        badguy_x -= 7


        # 6.3.1 castle을 공격할 경우 생명력이 감소하고 적들을 그리지 않는다.
        if badguy_x < 0 + castle.get_width():   # 적의 x좌표 < castle의 (x좌표 + 가로길이) 좌표
            player_health -= random.randint(5, 20)
            hit.play()  # 공격 받을 경우, 피격음 재생
            pass
        
        # 적들의 위치가 화면 밖을 벗어나면 그리지 않는다.
        elif badguy_x < -badguy.get_width():
            pass

        else:
            # 6.3.2 적과 화살의 충돌 처리
            # 적에 대한 사각형 객체 생성
            badguy_rect = pygame.Rect(badguy.get_rect())
            badguy_rect.left = badguy_x
            badguy_rect.top = badguy_y

            is_collide = False
            for _ in range(arrow_infos.qsize()):
                _, [arrow_x, arrow_y] = arrow_infos.get()
                # 화살에 대한 사각형 객체 생성
                arrow_rect = pygame.Rect(arrow.get_rect())
                arrow_rect.left = arrow_x
                arrow_rect.top = arrow_y
                # 적과 화살의 사각형 객체로 충돌 확인
                if badguy_rect.colliderect(arrow_rect):
                    # 명중 갯수 추가
                    accuracy[0] += 1
                    is_collide = True
                    enemy.play()    # 타격했을 경우, 적의 소리 재생
                else:
                    arrow_infos.put([_, [arrow_x, arrow_y]])

            # 화살에 맞지 않은 적들을 그린다.
            if not is_collide:
                screen.blit(badguy, [badguy_x, badguy_y])
                badguy_infos.put([badguy_x, badguy_y])
        

    # 6.4 글자를 그린다.
    font = pygame.font.Font(None, 24)
    remain_time = play_time - pygame.time.get_ticks()
    if remain_time < 0: remain_time = 0
    text_survived = font.render("{:d} : {:02d}".format(
        remain_time // 1000 // 60,    # 분
        remain_time // 1000 % 60),    # 초
        True, (0, 0, 0))
    text_rect = text_survived.get_rect()
    text_rect.topright = [635, 5]
    screen.blit(text_survived, text_rect)


    # 6.5 생명력 바를 그린다.
    screen.blit(healthbar, (5, 5))
    for i in range(player_health):
        screen.blit(health, [i + 8, 8])


    # 7. 화면을 다시 그린다.
    pygame.display.flip()


    # 8. 이벤트 검사
    for event in pygame.event.get():
        # X를 눌렀으면, 게임을 종료
        if event.type == pygame.QUIT:
            # 게임종료한다.
            pygame.quit()
            running = False
        
        # [W, A, S, D] 키를 눌렀을 때, 키 정보 갱신
        # 키 정보는 플레이어의 이동을 위해 사용됨 (# 9. 플레이어 이동)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                keys[0] = True
            elif event.key == pygame.K_a:
                keys[1] = True
            elif event.key == pygame.K_s:
                keys[2] = True
            elif event.key == pygame.K_d:
                keys[3] = True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_w:
                keys[0] = False
            elif event.key == pygame.K_a:
                keys[1] = False
            elif event.key == pygame.K_s:
                keys[2] = False
            elif event.key == pygame.K_d:
                keys[3] = False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            accuracy[1] += 1
            # 화살의 방항과 위치를 화살 정보에 추가한다.
            # 화살의 방향 : 플레이어의 위치부터 마우스 위치까지의 Θ값을 구한다.
            arrow_angle = math.atan2(mouse_pos[1] - (player_rot_pos[1] + player_rot.get_height() // 2), mouse_pos[0] - (player_rot_pos[0] + player_rot.get_width() // 2))
            # 화살이 발사되는 위치 (x, y)
            player_pos_center = [player_rot_pos[0] + player_rot.get_width() // 2, player_rot_pos[1] + player_rot.get_height() // 2]
            arrow_infos.put([arrow_angle, player_pos_center])

            shoot.play()    # 화살이 발사됐을 경우, 사격 소리 재생
    

    # 9. 플레이어 이동
    if keys[0]:     # W키가 눌린 경우,
        player_pos[1] -= 5  # 플레이어가 위로 5만큼 이동
    elif keys[2]:   # S키가 눌린 경우,
        player_pos[1] += 5  # 플레이어가 아래로 5만큼 이동
    
    if keys[1]:     # A키가 눌린 경우,
        player_pos[0] -= 5  # 플레이어가 왼쪽으로 5만큼 이동
    elif keys[3]:   # D키가 눌린 경우,
        player_pos[0] += 5  # 플레이어가 오른쪽으로 5만큼 이동

    
    # 10. 승리/패배 판정
    if pygame.time.get_ticks() >= play_time:
        running = False
    if player_health <= 0:
        running = False
        is_win = False


# 11. 승리/패배 결과 그리기
acc = accuracy[0] / accuracy[1] * 100 if accuracy[1] != 0 else 0

if is_win:
    pygame.font.init()
    font = pygame.font.Font(None, 24)
    text = font.render("Accuracy: {:f}%".format(acc), True, (0, 255, 0)) # 초록색
    text_rect = text.get_rect()
    text_rect.centerx = screen.get_rect().centerx
    text_rect.centery = screen.get_rect().centery + 24
    screen.blit(youwin, (0, 0))
    screen.blit(text, text_rect)
else:
    pygame.font.init()
    font = pygame.font.Font(None, 24)
    text = font.render("Accuracy: {:f}%".format(acc), True, (255, 0, 0)) # 빨간색
    text_rect = text.get_rect()
    text_rect.centerx = screen.get_rect().centerx
    text_rect.centery = screen.get_rect().centery + 24
    screen.blit(gameover, (0, 0))
    screen.blit(text, text_rect)

pygame.display.flip()

# x 버튼이 눌릴 때 까지 게임 창 대기
while True:
    for event in pygame.event.get():
        # X를 눌렀으면, 게임을 종료
        if event.type == pygame.QUIT:
            # 게임종료한다.
            pygame.quit()
            exit(0)
