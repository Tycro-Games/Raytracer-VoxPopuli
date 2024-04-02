#include "precomp.h"
#include "ModifyingProp.h"

ModifyingProp::ModifyingProp(Scene& scene, float time, uint32_t startingIndex)
  : toChange(time), voxelVolume(scene), index(startingIndex)

{
  timer = make_unique<Timer>();
}

void ModifyingProp::Update(float /*deltaTime*/)
{
  if (timer->elapsed() > toChange)
  {
    timer->reseting();
    //voxelVolume.ResetGrid();
    voxelVolume.LoadModelPartial("assets/monu2.vox", index);
    index += 13;
    if (index > 64)
      index = 13;
  }
}

bool ModifyingProp::GetUpdate()
{
  return timer->elapsed() < 0.1f;
}
