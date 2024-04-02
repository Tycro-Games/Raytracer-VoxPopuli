#include "precomp.h"
#include "ModifyingProp.h"

ModifyingProp::ModifyingProp(Scene& scene, float time, uint32_t startingIndex, uint32_t increasingRate)
  : toChange(time), voxelVolume(scene), increaseRate(increasingRate), index(startingIndex)

{
  timer = make_unique<Timer>();
}

void ModifyingProp::Update(float /*deltaTime*/)
{
  if (timer->elapsed() > toChange)
  {
    timer->reseting();
    //voxelVolume.ResetGrid();
    voxelVolume.LoadModelPartial("assets/monu2.vox", index, increaseRate);
    index += increaseRate;
    if (index > 64)
      index = increaseRate;
  }
}

bool ModifyingProp::GetUpdate()
{
  return timer->elapsed() < 0.1f;
}
